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


def _capture_region_bgr(region: tuple[int, int, int, int]) -> np.ndarray:
    shot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def _run_ocr_on_bgr(
    img_bgr: np.ndarray,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, img_bgr)
        return _run_ocr_on_file(tmp_path, debug_dir=debug_dir, color_config=color_config)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _estimate_vertical_shift(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    if prev_bgr.shape != curr_bgr.shape:
        return 9999.0

    h, w = prev_bgr.shape[:2]
    x0, x1 = int(w * 0.1), int(w * 0.9)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    if x1 <= x0 or y1 <= y0:
        return 9999.0

    prev_gray = cv2.cvtColor(prev_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    (dx, dy), response = cv2.phaseCorrelate(prev_gray, curr_gray)
    dy_abs = abs(float(dy))

    if np.isnan(dy_abs):
        return 0.0

    # Low confidence phase correlation falls back to average pixel delta.
    if response < 0.02:
        prev_u8 = prev_gray.astype(np.uint8)
        curr_u8 = curr_gray.astype(np.uint8)
        mean_diff = float(np.mean(cv2.absdiff(prev_u8, curr_u8)))
        if mean_diff >= 2.0:
            return 9999.0
        return 0.0

    return dy_abs


def _best_vertical_overlap(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> tuple[str, int, float]:
    h, w = prev_bgr.shape[:2]
    if curr_bgr.shape[:2] != (h, w):
        return ("append_bottom", 0, 9999.0)
    if h < 40 or w < 40:
        return ("append_bottom", 0, 9999.0)

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    target_w = min(360, w)
    scale = target_w / float(w)
    target_h = max(1, int(h * scale))
    prev_small = cv2.resize(prev_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(curr_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    min_ov = max(12, int(target_h * 0.2))
    max_ov = max(min_ov, int(target_h * 0.92))
    step = 4

    best_mode = "append_bottom"
    best_ov = 0
    best_score = 9999.0

    for ov in range(max_ov, min_ov - 1, -step):
        a1 = prev_small[-ov:, :]
        b1 = curr_small[:ov, :]
        score1 = float(np.mean(cv2.absdiff(a1, b1)))
        if score1 < best_score:
            best_score = score1
            best_mode = "append_bottom"
            best_ov = ov

        a2 = prev_small[:ov, :]
        b2 = curr_small[-ov:, :]
        score2 = float(np.mean(cv2.absdiff(a2, b2)))
        if score2 < best_score:
            best_score = score2
            best_mode = "prepend_top"
            best_ov = ov

    if best_score > 26.0:
        return ("append_bottom", 0, best_score)

    ov_full = int(best_ov / scale) if scale > 0 else 0
    ov_full = max(0, min(ov_full, h - 1))
    return (best_mode, ov_full, best_score)


def stitch_scrolled_captures(captures: list[np.ndarray]) -> np.ndarray:
    if not captures:
        raise ValueError("captures is empty")
    if len(captures) == 1:
        return captures[0].copy()

    stitched = captures[0].copy()
    for curr in captures[1:]:
        mode, overlap, _ = _best_vertical_overlap(stitched[-curr.shape[0]:, :], curr)
        if mode == "append_bottom":
            if overlap > 0:
                stitched = np.vstack([stitched, curr[overlap:, :]])
            else:
                stitched = np.vstack([stitched, curr])
        else:
            if overlap > 0:
                stitched = np.vstack([curr[:-overlap, :], stitched])
            else:
                stitched = np.vstack([curr, stitched])
    return stitched


def _overlap_ratio(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """Return matched vertical overlap as a ratio of frame height (0~1)."""
    h = prev_bgr.shape[0]
    if h <= 0:
        return 0.0
    _, overlap_px, score = _best_vertical_overlap(prev_bgr, curr_bgr)
    if score > 26.0 or overlap_px <= 0:
        return 0.0
    return float(overlap_px) / float(h)


def scroll_capture_images(
    window,
    rounds: int,
    pause: float,
    region: tuple[int, int, int, int] | None = None,
    progress_callback=None,
    shift_ratio: float = 0.45,
    min_overlap_ratio: float = 0.25,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Capture scrolled frames with forced overlap.
    Target shift defaults to ~45% viewport so neighboring frames keep ~55% overlap.
    """
    if region is None:
        region = _chat_region(window)

    cx = region[0] + region[2] // 2
    cy = region[1] + region[3] // 2
    viewport_h = max(1.0, float(region[3]))
    shift_ratio = max(0.2, min(0.7, float(shift_ratio)))
    min_overlap_ratio = max(0.15, min(0.6, float(min_overlap_ratio)))
    target_shift_px = max(1.0, viewport_h * shift_ratio)

    pyautogui.click(cx, cy)
    time.sleep(0.3)

    captures: list[np.ndarray] = []
    overlap_ratios: list[float] = []
    stopped_reason = "completed"

    def _scroll_to_target(
        start_img: np.ndarray,
        target_shift: float,
        *,
        direction: int = 1,
        initial_step: int | None = None,
    ) -> tuple[bool, np.ndarray, float]:
        img_prev = start_img
        img_last = start_img
        moved_px = 0.0
        no_move_streak = 0

        # Conservative wheel steps: avoid jumping more than one viewport.
        step_units = initial_step if initial_step is not None else max(20, int(region[3] / 20))
        min_step_units = 12
        max_step_units = max(80, int(region[3] / 4))
        max_scroll_attempts = 16
        sign = 1 if direction >= 0 else -1

        pyautogui.moveTo(cx, cy)
        for _ in range(max_scroll_attempts):
            if moved_px >= target_shift:
                break

            pyautogui.scroll(sign * step_units)
            time.sleep(max(0.03, pause / 6))
            img_now = _capture_region_bgr(region)
            img_last = img_now
            shift_y = _estimate_vertical_shift(img_prev, img_now)

            if shift_y < 1.5:
                no_move_streak += 1
                step_units = min(max_step_units, max(min_step_units, int(step_units * 1.35)))
                if no_move_streak >= 2:
                    break
                continue

            no_move_streak = 0
            moved_px += shift_y
            remaining = target_shift - moved_px
            if remaining <= 0:
                img_prev = img_now
                break

            gain = shift_y / max(step_units, 1)
            desired = int(remaining / max(gain, 0.08))
            step_units = max(min_step_units, min(max_step_units, desired))
            img_prev = img_now

        return (moved_px >= 3.0), img_last, moved_px

    def _ensure_overlap(prev_img: np.ndarray, curr_img: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        Gate neighboring frames by minimum overlap ratio.
        If overlap is too small (overscrolled), nudge back and recapture.
        """
        ratio = _overlap_ratio(prev_img, curr_img)
        if ratio >= min_overlap_ratio:
            return curr_img, ratio, True

        corrected = curr_img
        for attempt in range(4):
            # Overscrolled: move opposite direction with tiny steps.
            back_target = max(8.0, viewport_h * (min_overlap_ratio - ratio + 0.08))
            tiny_step = max(10, int(region[3] / 40))
            moved, corrected, _ = _scroll_to_target(
                corrected,
                back_target,
                direction=-1,
                initial_step=tiny_step,
            )
            ratio = _overlap_ratio(prev_img, corrected)
            if ratio >= min_overlap_ratio:
                return corrected, ratio, True
            if not moved:
                break
            print(
                f"overlap gate retry {attempt + 1}/4: ratio={ratio:.3f}, "
                f"need>={min_overlap_ratio:.3f}"
            )

        return corrected, ratio, ratio >= min_overlap_ratio

    next_capture: np.ndarray | None = None
    for i in range(rounds):
        if progress_callback:
            progress_callback(i + 1, rounds)

        curr_capture = next_capture if next_capture is not None else _capture_region_bgr(region)
        captures.append(curr_capture)
        if i >= rounds - 1:
            break

        moved, candidate, moved_px = _scroll_to_target(curr_capture, target_shift_px, direction=1)
        if not moved:
            stopped_reason = f"scroll_ineffective(moved={moved_px:.2f}px)"
            print(f"scroll no longer effective (moved={moved_px:.2f}px), stop scrolling")
            break

        candidate, ratio, ok = _ensure_overlap(curr_capture, candidate)
        overlap_ratios.append(ratio)
        if not ok:
            stopped_reason = f"overlap_insufficient(ratio={ratio:.3f})"
            print(
                f"overlap insufficient ({ratio:.3f} < {min_overlap_ratio:.3f}), "
                "stop scrolling to avoid skipped messages"
            )
            # Keep the last candidate only if it still moved; otherwise discard.
            if ratio > 0.05:
                captures.append(candidate)
            break

        next_capture = candidate
        time.sleep(max(0.03, pause / 4))

    avg_overlap = float(sum(overlap_ratios) / len(overlap_ratios)) if overlap_ratios else 0.0
    stats = {
        "capture_count": len(captures),
        "avg_overlap_ratio": avg_overlap,
        "overlap_ratios": overlap_ratios,
        "shift_ratio": shift_ratio,
        "min_overlap_ratio": min_overlap_ratio,
        "stopped_reason": stopped_reason,
    }
    return captures, stats


def scroll_and_collect_frames(
    window,
    rounds: int,
    pause: float,
    region: tuple[int, int, int, int] | None = None,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
    progress_callback=None,
    shift_ratio: float = 0.45,
    min_overlap_ratio: float = 0.25,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """
    Primary scroll path: capture with overlap, OCR each frame, return frames for message merge.
    Stitched image is saved for debug only.
    """
    captures, capture_stats = scroll_capture_images(
        window,
        rounds=rounds,
        pause=pause,
        region=region,
        progress_callback=progress_callback,
        shift_ratio=shift_ratio,
        min_overlap_ratio=min_overlap_ratio,
    )
    if not captures:
        return [], {**capture_stats, "merged_count": 0}

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(captures, start=1):
            cv2.imwrite(str(Path(debug_dir) / f"capture_{idx:03d}.png"), img)
        try:
            stitched = stitch_scrolled_captures(captures)
            cv2.imwrite(str(Path(debug_dir) / "stitched.png"), stitched)
        except Exception as e:
            print(f"warning: stitch debug image skipped — {e}")

    frames: list[list[dict[str, Any]]] = []
    for idx, img in enumerate(captures, start=1):
        frame_debug = str(Path(debug_dir) / f"frame_{idx:03d}_ocr") if debug_dir else None
        rows = _run_ocr_on_bgr(img, debug_dir=frame_debug, color_config=color_config)
        frames.append(rows)
        if progress_callback:
            progress_callback(idx, len(captures))

    merged = merge_scrolled_frames(frames)
    stats = {
        **capture_stats,
        "merged_count": len(merged),
        "per_frame_counts": [len(f) for f in frames],
    }
    return frames, stats


def scroll_and_collect_stitched(
    window,
    rounds: int,
    pause: float,
    region: tuple[int, int, int, int] | None = None,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
    progress_callback=None,
    shift_ratio: float = 0.45,
    min_overlap_ratio: float = 0.25,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Compatibility wrapper: returns merged rows from per-frame OCR path."""
    frames, stats = scroll_and_collect_frames(
        window,
        rounds=rounds,
        pause=pause,
        region=region,
        debug_dir=debug_dir,
        color_config=color_config,
        progress_callback=progress_callback,
        shift_ratio=shift_ratio,
        min_overlap_ratio=min_overlap_ratio,
    )
    merged = merge_scrolled_frames(frames)
    return merged, int(stats.get("capture_count", 0)), stats


def scroll_and_collect(
    window,
    rounds: int,
    pause: float,
    region: tuple[int, int, int, int] | None = None,
    debug_dir: str | None = None,
    color_config: dict[str, Any] | None = None,
    progress_callback=None,
    shift_ratio: float = 0.45,
    min_overlap_ratio: float = 0.25,
) -> list[list[dict[str, Any]]]:
    frames, stats = scroll_and_collect_frames(
        window,
        rounds=rounds,
        pause=pause,
        region=region,
        debug_dir=debug_dir,
        color_config=color_config,
        progress_callback=progress_callback,
        shift_ratio=shift_ratio,
        min_overlap_ratio=min_overlap_ratio,
    )
    print(
        f"captured {stats.get('capture_count', 0)} frames, "
        f"avg_overlap={stats.get('avg_overlap_ratio', 0):.3f}, "
        f"merged={stats.get('merged_count', 0)}, "
        f"stop={stats.get('stopped_reason')}"
    )
    return frames


def merge_scrolled_frames(frames: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Merge per-frame OCR results in chat chronological order (top=older → bottom=newer).

    Supports both scroll directions:
    - scroll down / content up: new frame overlaps at bottom of previous → append
    - scroll up / older history: new frame overlaps at top of previous → prepend
    """
    merged: list[dict[str, Any]] = []
    for rows in frames:
        normalized_rows = [
            {
                "sender": row["sender"],
                "content": row["content"],
                "confidence": row.get("confidence"),
            }
            for row in rows
            if row.get("content")
        ]
        if not normalized_rows:
            continue
        if not merged:
            merged = normalized_rows
            continue

        mode, overlap = _find_best_overlap(merged, normalized_rows)
        if mode == "append":
            merged.extend(normalized_rows[overlap:])
        elif mode == "prepend":
            merged = normalized_rows[: len(normalized_rows) - overlap] + merged
        else:
            # No reliable contiguous overlap: fall back to message-level inflation control.
            merged = _merge_by_unique_keys(merged, normalized_rows)

    return _dedupe_near_repeats(_collapse_adjacent_duplicates(merged))


def _message_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("sender", "")), _normalize_message_text(row.get("content", "")))


def _texts_equivalent(a: str, b: str) -> bool:
    na = _normalize_message_text(a)
    nb = _normalize_message_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Tolerate minor OCR drift on the same bubble across frames.
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 4 and shorter in longer and len(shorter) / len(longer) >= 0.75:
        return True
    return False


def _rows_match(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.get("sender") != b.get("sender"):
            return False
        if not _texts_equivalent(str(a.get("content", "")), str(b.get("content", ""))):
            return False
    return True


def _find_overlap_size(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    max_window: int = 20,
) -> int:
    """Longest size where left[-size:] matches right[:size]."""
    if not left or not right:
        return 0
    max_overlap = min(len(left), len(right), max_window)
    for size in range(max_overlap, 0, -1):
        if _rows_match(left[-size:], right[:size]):
            return size
    return 0


def _find_best_overlap(
    merged: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    max_window: int = 20,
) -> tuple[str, int]:
    """
    Return (mode, overlap_size):
    - append: merged tail == rows head  (newer content below)
    - prepend: rows tail == merged head (older content above)
    - none: no reliable overlap
    """
    append_ov = _find_overlap_size(merged, rows, max_window=max_window)
    prepend_ov = _find_overlap_size(rows, merged, max_window=max_window)

    # Prefer the larger overlap; require at least 1 matching message.
    if append_ov == 0 and prepend_ov == 0:
        return ("none", 0)
    if prepend_ov > append_ov:
        return ("prepend", prepend_ov)
    if append_ov > prepend_ov:
        return ("append", append_ov)
    # Tie: WeChat history scroll usually reveals older messages (prepend).
    return ("prepend", prepend_ov)


def _merge_by_unique_keys(
    merged: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Fallback when contiguous overlap fails.
    Keep chronological order by prepending unseen older messages and appending unseen newer ones.
    """
    merged_keys = {_message_key(r) for r in merged}
    # Prefer treating unmatched frame as older history (common when scrolling up).
    new_older = [r for r in rows if _message_key(r) not in merged_keys]
    if not new_older:
        return merged

    # Decide side by comparing first/last key proximity.
    # If frame's last messages look like merged's first → prepend.
    # If frame's first messages look like merged's last → append.
    head_hit = 0
    tail_hit = 0
    probe = min(3, len(rows), len(merged))
    for i in range(probe):
        if _rows_match([rows[-(i + 1)]], [merged[i]]):
            head_hit += 1
        if _rows_match([rows[i]], [merged[-(i + 1)]]):
            tail_hit += 1

    if head_hit >= tail_hit:
        return new_older + merged
    return merged + new_older


def _collapse_adjacent_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    out = [rows[0]]
    for row in rows[1:]:
        prev = out[-1]
        if prev.get("sender") == row.get("sender") and _texts_equivalent(
            str(prev.get("content", "")),
            str(row.get("content", "")),
        ):
            # Keep higher confidence copy.
            prev_conf = float(prev.get("confidence") or 0.0)
            row_conf = float(row.get("confidence") or 0.0)
            if row_conf > prev_conf:
                out[-1] = row
            continue
        out.append(row)
    return out


def _dedupe_near_repeats(rows: list[dict[str, Any]], window: int = 20) -> list[dict[str, Any]]:
    """
    Drop overlap leftovers that reappear shortly after an earlier copy.
    Keeps intentional repeats that are far apart.
    """
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        key = _message_key(row)
        recent = {_message_key(r) for r in out[-window:]}
        if key in recent:
            continue
        out.append(row)
    return out


def _normalize_message_text(text: str) -> str:
    return "".join(str(text).split())


def export_txt(frames: list[list[dict[str, Any]]], path: str, meta: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 微信聊天OCR导出记录\n")
        f.write(f"# 生成时间(UTC): {meta.get('generated_utc')}\n")
        f.write(f"# 采集屏数: {len(frames)}\n\n")
        all_messages = merge_scrolled_frames(frames)
        for msg in all_messages:
            f.write(f"[{msg['sender']}]: {msg['content']}\n")


def export_json(frames: list[list[dict[str, Any]]], path: str, meta: dict[str, Any]) -> None:
    all_messages = merge_scrolled_frames(frames)
    payload = {"meta": meta, "messages": all_messages}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
