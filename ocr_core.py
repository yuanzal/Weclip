from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import tkinter as tk
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from paddleocr import PaddleOCR

# 解决Paddle 3.x CPU环境oneDNN报错
os.environ.setdefault("FLAGS_use_mkldnn", "0")

# 初始化OCR模型（仅首次运行下载模型）
ocr = PaddleOCR(
    lang="ch",
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    enable_mkldnn=False,
)

# 安全设置：防止误操作失控
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _hsv_bounds(
    center_hsv: tuple[int, int, int],
    h_tol: int,
    s_tol: int,
    v_tol: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    h, s, v = center_hsv
    low = (_clamp(h - h_tol, 0, 179), _clamp(s - s_tol, 0, 255), _clamp(v - v_tol, 0, 255))
    high = (_clamp(h + h_tol, 0, 179), _clamp(s + s_tol, 0, 255), _clamp(v + v_tol, 0, 255))
    return low, high


# ocr_core.py 中的核心修改

def get_wechat_window(title: str = "微信"):
    """获取微信窗口，仅尝试激活，不强制阻塞"""
    try:
        wins = gw.getWindowsWithTitle(title)
        if not wins:
            return None

        win = wins[0]
        # 尝试恢复并激活，但如果失败（例如权限问题）不应抛出异常崩掉整个程序
        try:
            if win.isMinimized:
                win.restore()
            win.activate()
        except:
            pass
        return win
    except Exception:
        return None

def _chat_region(window) -> tuple[int, int, int, int]:
    left, top, w, h = window.left, window.top, window.width, window.height
    sidebar = 280
    top_bar = 80
    bottom_margin = 100
    return (
        left + sidebar,
        top + top_bar,
        max(1, w - sidebar),
        max(1, h - top_bar - bottom_margin),
    )


def select_region_interactive() -> tuple[int, int, int, int] | None:
    result: list[tuple[int, int, int, int] | None] = [None]
    root = tk.Tk()
    root.title("")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.35)
    root.configure(bg="black")
    root.overrideredirect(True)

    canvas = tk.Canvas(root, highlightthickness=0, bg="black", cursor="crosshair")
    canvas.pack(fill=tk.BOTH, expand=True)
    hint = tk.Label(
        root,
        text="拖动鼠标框选聊天识别区域  ·  Esc 取消",
        fg="white",
        bg="#333333",
        font=("Microsoft YaHei UI", 12),
        padx=12,
        pady=8,
    )
    hint.place(relx=0.5, y=24, anchor=tk.N)
    start: dict[str, int] = {}

    def on_press(e: tk.Event) -> None:
        start["x"] = e.x
        start["y"] = e.y
        canvas.delete("sel")

    def on_drag(e: tk.Event) -> None:
        if "x" not in start:
            return
        canvas.delete("sel")
        x0, y0 = start["x"], start["y"]
        canvas.create_rectangle(x0, y0, e.x, e.y, outline="#ff4444", width=2, tags="sel")

    def on_release(e: tk.Event) -> None:
        if "x" not in start:
            return
        x0, y0 = start["x"], start["y"]
        x1, y1 = e.x, e.y
        rx = canvas.winfo_rootx()
        ry = canvas.winfo_rooty()
        left = rx + min(x0, x1)
        top = ry + min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if width >= 2 and height >= 2:
            result[0] = (left, top, width, height)
            root.quit()
        else:
            canvas.delete("sel")
            start.clear()

    def on_escape(_: tk.Event) -> None:
        result[0] = None
        root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)
    root.focus_force()
    root.mainloop()
    root.destroy()
    return result[0]


def _build_color_masks(
    img_bgr: np.ndarray,
    color_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color_config and color_config.get("other_hsv") and color_config.get("self_hsv"):
        h_tol = int(color_config.get("h_tol", 12))
        s_tol = int(color_config.get("s_tol", 60))
        v_tol = int(color_config.get("v_tol", 60))
        other_low, other_high = _hsv_bounds(tuple(color_config["other_hsv"]), h_tol, s_tol, v_tol)
        self_low, self_high = _hsv_bounds(tuple(color_config["self_hsv"]), h_tol, s_tol, v_tol)
        white_mask = cv2.inRange(hsv, other_low, other_high)
        green_mask = cv2.inRange(hsv, self_low, self_high)
    else:
        # 默认对方气泡 HSV（用户指定）
        default_other_hsv = (120, 2, 240)
        other_low, other_high = _hsv_bounds(default_other_hsv, h_tol=30, s_tol=40, v_tol=30)
        white_mask = cv2.inRange(hsv, other_low, other_high)
        green_mask = cv2.inRange(hsv, (35, 45, 80), (90, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    return white_mask, green_mask


def _extract_bubbles_from_mask(mask: np.ndarray, sender: str, image_w: int, image_h: int) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles: list[dict[str, Any]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 900 or w < 70 or h < 26:
            continue
        if w > image_w * 0.92 or h > image_h * 0.4:
            continue
        if x <= 2 or y <= 2 or x + w >= image_w - 2:
            continue
        pad = 6
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_w, x + w + pad)
        y1 = min(image_h, y + h + pad)
        bubbles.append({"sender": sender, "bbox": (x0, y0, x1, y1), "top_y": y0})
    return bubbles


def _save_debug_images(
    debug_dir: str,
    img: np.ndarray,
    white_mask: np.ndarray,
    green_mask: np.ndarray,
    bubbles: list[dict[str, Any]],
) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(str(Path(debug_dir) / "00_source.png"), img)
    cv2.imwrite(str(Path(debug_dir) / "01_white_mask.png"), white_mask)
    cv2.imwrite(str(Path(debug_dir) / "02_green_mask.png"), green_mask)
    annotated = img.copy()
    for b in bubbles:
        x0, y0, x1, y1 = b["bbox"]
        color = (255, 0, 0) if b["sender"] == "对方" else (0, 180, 0)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        cv2.putText(annotated, b["sender"], (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(Path(debug_dir) / "03_bubbles.png"), annotated)


def _detect_color_bubbles(
    path: str,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取截图文件：{path}")
    h, w = img.shape[:2]
    white_mask, green_mask = _build_color_masks(img, color_config=color_config)
    white_bubbles = _extract_bubbles_from_mask(white_mask, "对方", w, h)
    green_bubbles = _extract_bubbles_from_mask(green_mask, "自己", w, h)
    bubbles = sorted([*white_bubbles, *green_bubbles], key=lambda x: x["top_y"])
    debug_stats = {
        "image_size": {"width": w, "height": h},
        "mask_pixels": {"white": int(np.count_nonzero(white_mask)), "green": int(np.count_nonzero(green_mask))},
        "bubble_count": {"white": len(white_bubbles), "green": len(green_bubbles), "total": len(bubbles)},
    }
    if debug_dir:
        _save_debug_images(debug_dir, img, white_mask, green_mask, bubbles)
    return img, bubbles, debug_stats


def _run_ocr_on_file(
    path: str,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    img_bgr, bubbles, debug_stats = _detect_color_bubbles(path, debug_dir=debug_dir, color_config=color_config)
    out: list[dict[str, Any]] = []
    debug_log: dict[str, Any] = {"stats": debug_stats, "bubbles": [], "result_count": 0}
    if not bubbles:
        if debug_dir:
            debug_log["note"] = "没有检测到任何白色/绿色气泡。"
            with open(Path(debug_dir) / "debug_log.json", "w", encoding="utf-8") as f:
                json.dump(debug_log, f, ensure_ascii=False, indent=2)
            print(f"🧪 调试输出已保存: {os.path.abspath(debug_dir)}")
        return out

    emoji_filter = {"😂", "😅", "🤣", "👍", "😊", "😁", "😆", "🥰", "😍", "😎", "🤩"}
    for idx, bubble in enumerate(bubbles, start=1):
        x0, y0, x1, y1 = bubble["bbox"]
        crop = img_bgr[y0:y1, x0:x1]
        bubble_log: dict[str, Any] = {"id": idx, "sender": bubble["sender"], "bbox": [x0, y0, x1, y1], "status": "pending"}
        if crop.size == 0:
            bubble_log["status"] = "skip_empty_crop"
            debug_log["bubbles"].append(bubble_log)
            continue
        if debug_dir:
            cv2.imwrite(str(Path(debug_dir) / f"crop_{idx:03d}_{bubble['sender']}.png"), crop)
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, crop)
            results = ocr.predict(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if not results:
            bubble_log["status"] = "skip_no_ocr_result"
            debug_log["bubbles"].append(bubble_log)
            continue
        res = results[0]
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        bubble_log["ocr_texts"] = [str(t[0] if isinstance(t, tuple) else t) for t in texts]
        bubble_log["ocr_scores"] = [float(s) for s in scores]
        merged_parts: list[str] = []
        valid_scores: list[float] = []
        dropped_parts: list[dict[str, Any]] = []
        for text, conf in zip(texts, scores):
            if isinstance(text, tuple):
                text = text[0]
            text_str = str(text).strip()
            conf_float = float(conf)
            if conf_float < 0.6 or not text_str:
                dropped_parts.append({"text": text_str, "score": conf_float, "reason": "low_conf_or_empty"})
                continue
            if all(c in emoji_filter for c in text_str):
                dropped_parts.append({"text": text_str, "score": conf_float, "reason": "emoji_only"})
                continue
            merged_parts.append(text_str)
            valid_scores.append(conf_float)
        if not merged_parts:
            bubble_log["status"] = "skip_all_filtered"
            bubble_log["dropped_parts"] = dropped_parts
            debug_log["bubbles"].append(bubble_log)
            continue
        bubble_log["status"] = "accepted"
        bubble_log["accepted_text"] = "".join(merged_parts)
        bubble_log["accepted_score_min"] = min(valid_scores)
        if dropped_parts:
            bubble_log["dropped_parts"] = dropped_parts
        debug_log["bubbles"].append(bubble_log)
        out.append({"top_y": bubble["top_y"], "sender": bubble["sender"], "content": "".join(merged_parts), "confidence": min(valid_scores)})
    out.sort(key=lambda x: x["top_y"])
    for r in out:
        r.pop("top_y", None)
    if debug_dir:
        debug_log["result_count"] = len(out)
        with open(Path(debug_dir) / "debug_log.json", "w", encoding="utf-8") as f:
            json.dump(debug_log, f, ensure_ascii=False, indent=2)
        print(f"🧪 调试输出已保存: {os.path.abspath(debug_dir)}")
    return out


def ocr_chat_region(
    window,
    image_path: str | None = None,
    region: tuple[int, int, int, int] | None = None,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if region is None:
        region = _chat_region(window)
    shot = pyautogui.screenshot(region=region)
    path = image_path
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            shot.save(path)
            return _run_ocr_on_file(path, debug_dir=debug_dir, color_config=color_config)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    shot.save(path)
    return _run_ocr_on_file(path, debug_dir=debug_dir, color_config=color_config)


def _frame_fingerprint(rows: list[dict[str, Any]]) -> str:
    raw = "\n".join(f"{r['sender']}|{r['content']}" for r in rows)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def scroll_and_collect(
    window,
    rounds: int,
    pause: float,
    region: tuple[int, int, int, int] | None = None,
    debug_dir: str | None = None,
) -> list[list[dict[str, Any]]]:
    if region is None:
        region = _chat_region(window)
    cx = region[0] + region[2] // 2
    cy = region[1] + region[3] // 2
    pyautogui.click(cx, cy)
    time.sleep(0.2)
    frames: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for i in range(rounds):
        frame_debug_dir = str(Path(debug_dir) / f"frame_{i + 1:03d}") if debug_dir else None
        rows = ocr_chat_region(window, region=region, debug_dir=frame_debug_dir)
        fp = _frame_fingerprint(rows)
        if fp in seen:
            print(f"✅ 第 {i + 1} 屏与已有内容重复，停止滚动。")
            break
        seen.add(fp)
        frames.append(rows)
        print(f"✅ 已采集第 {i + 1}/{rounds} 屏，本屏 {len(rows)} 条有效聊天。")
        pyautogui.scroll(8)
        time.sleep(pause)
    return frames


def export_txt(frames: list[list[dict[str, Any]]], path: str, meta: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 微信聊天OCR导出记录\n")
        f.write(f"# 生成时间(UTC): {meta.get('generated_utc')}\n")
        f.write(f"# 采集屏数: {len(frames)}\n\n")
        all_messages: list[dict[str, Any]] = []
        for rows in frames:
            all_messages.extend(rows)
        for msg in all_messages:
            f.write(f"[{msg['sender']}]: {msg['content']}\n")


def export_json(frames: list[list[dict[str, Any]]], path: str, meta: dict[str, Any]) -> None:
    all_messages: list[dict[str, Any]] = []
    for rows in frames:
        all_messages.extend(rows)
    payload = {"meta": meta, "messages": all_messages}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
