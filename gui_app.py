from __future__ import annotations

import os
import tempfile
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageTk

from ocr_core import (
    _build_color_masks,
    _extract_bubbles_from_mask,
    _run_ocr_on_file,
    export_json,
    export_txt,
    get_wechat_window,
    select_region_interactive,
)


class OcrTunerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("WeChat OCR 调参工具")
        self.root.geometry("1480x920")
        self.root.minsize(1320, 820)
        self.root.configure(bg="#f5f9ff")
        self.ui_font_family = self._pick_font_family()
        self.root.option_add("*Font", (self.ui_font_family, 11))

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f5f9ff")
        self.style.configure("TLabel", background="#f5f9ff", foreground="#1f2d3d")
        self.style.configure("TLabelframe", background="#f5f9ff", foreground="#2b4b7c", bordercolor="#b9d4f7")
        self.style.configure("TLabelframe.Label", background="#f5f9ff", foreground="#2b4b7c")
        self.style.configure("TButton", padding=7)
        self.style.configure("TCheckbutton", background="#f5f9ff", foreground="#1f2d3d")

        self.base_img: np.ndarray | None = None
        self.preview_imgtk: ImageTk.PhotoImage | None = None
        self.scale_ratio = 1.0
        self.view_offset = (0, 0)
        self.pick_target: str | None = None

        self.other_hsv: tuple[int, int, int] | None = None
        self.self_hsv: tuple[int, int, int] | None = None
        self.h_tol = tk.IntVar(value=12)
        self.s_tol = tk.IntVar(value=60)
        self.v_tol = tk.IntVar(value=60)
        self.save_debug = tk.BooleanVar(value=True)
        self.out_path = tk.StringVar(value=str(Path.cwd() / "wechat_export_gui.txt"))
        self.debug_dir = tk.StringVar(value=str(Path.cwd() / "debug_gui"))
        self.window_title = tk.StringVar(value="微信")
        self.status = tk.StringVar(value="1) 截图或加载图片  2) 采样对方/自己颜色  3) 预览  4) 导出")
        self._build_layout()

    def run(self) -> None:
        self.root.mainloop()

    def _pick_font_family(self) -> str:
        families = set(tkfont.families())
        for name in ("Source Han Sans CN", "Source Han Sans SC", "思源黑体", "Microsoft YaHei UI"):
            if name in families:
                return name
        return "TkDefaultFont"

    def _create_round_button(self, parent, text: str, command, accent: bool = False, min_width: int = 128) -> tk.Canvas:
        bg = "#2f80ff" if accent else "#e8f1ff"
        fg = "#ffffff" if accent else "#1e4f9a"
        border = "#2f80ff" if accent else "#9ec5ff"
        font_obj = tkfont.Font(family=self.ui_font_family, size=11)
        btn_w = max(min_width, font_obj.measure(text) + 44)
        canvas = tk.Canvas(parent, width=btn_w, height=38, bg="#f5f9ff", highlightthickness=0, bd=0)
        r = 10
        x1, y1 = btn_w - 2, 36
        points = [2 + r, 2, x1 - r, 2, x1, 2, x1, 2 + r, x1, y1 - r, x1, y1, x1 - r, y1, 2 + r, y1, 2, y1, 2, y1 - r, 2, 2 + r, 2, 2]
        canvas.create_polygon(points, smooth=True, splinesteps=18, fill=bg, outline=border, width=1, tags="btn")
        canvas.create_text(btn_w // 2, 19, text=text, fill=fg, font=(self.ui_font_family, 11), tags="txt")
        canvas.bind("<Button-1>", lambda _: command())
        canvas.bind("<Enter>", lambda _: canvas.itemconfig("btn", fill="#1f6feb" if accent else "#dcecff"))
        canvas.bind("<Leave>", lambda _: canvas.itemconfig("btn", fill=bg))
        canvas.configure(cursor="hand2")
        return canvas

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root)
        root_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        root_frame.columnconfigure(0, weight=3)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(0, weight=1)
        left = ttk.Frame(root_frame)
        right = ttk.Frame(root_frame)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        toolbar = ttk.Frame(left)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        self._create_round_button(toolbar, "框选截图", self.capture_region).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "加载图片", self.load_image).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "选取对方色", lambda: self.start_pick("other")).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "选取自己色", lambda: self.start_pick("self")).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "预览掩码/气泡框", self.preview_masks, min_width=220).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(left, bg="#ffffff", highlightthickness=1, highlightbackground="#c9defa")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        ttk.Label(left, textvariable=self.status, anchor="w").pack(fill=tk.X, pady=(8, 0))

        lf_color = ttk.LabelFrame(right, text="颜色与容差")
        lf_color.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(lf_color, text="对方HSV").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.lbl_other = tk.Label(lf_color, text="未采样（将使用默认颜色，可能影响识别效果）", fg="#d4380d", bg="#f5f9ff", font=(self.ui_font_family, 11))
        self.lbl_other.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(lf_color, text="自己HSV").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.lbl_self = tk.Label(lf_color, text="未采样（将使用默认颜色，可能影响识别效果）", fg="#d4380d", bg="#f5f9ff", font=(self.ui_font_family, 11))
        self.lbl_self.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(lf_color, text="H 容差").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(lf_color, from_=2, to=40, orient=tk.HORIZONTAL, variable=self.h_tol).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Label(lf_color, text="S 容差").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(lf_color, from_=10, to=120, orient=tk.HORIZONTAL, variable=self.s_tol).grid(row=3, column=1, sticky="ew", padx=6)
        ttk.Label(lf_color, text="V 容差").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(lf_color, from_=10, to=120, orient=tk.HORIZONTAL, variable=self.v_tol).grid(row=4, column=1, sticky="ew", padx=6)
        lf_color.columnconfigure(1, weight=1)

        lf_out = ttk.LabelFrame(right, text="输出设置")
        lf_out.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(lf_out, text="微信窗口标题").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(lf_out, textvariable=self.window_title).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Label(lf_out, text="导出文件").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(lf_out, textvariable=self.out_path).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(lf_out, text="选择", command=self.choose_out_path).grid(row=1, column=2, padx=6, pady=6)
        self.save_debug_toggle = tk.Checkbutton(
            lf_out, text="调试输出：开启", variable=self.save_debug, indicatoron=False, relief=tk.FLAT, bd=0,
            bg="#2f80ff", fg="#ffffff", activebackground="#5a9dff", activeforeground="#ffffff",
            selectcolor="#2f80ff", padx=12, pady=5, font=(self.ui_font_family, 11), command=self._on_debug_toggle,
        )
        self.save_debug_toggle.grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(lf_out, text="调试目录").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(lf_out, textvariable=self.debug_dir).grid(row=3, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(lf_out, text="选择", command=self.choose_debug_dir).grid(row=3, column=2, padx=6, pady=6)
        lf_out.columnconfigure(1, weight=1)

        lf_act = ttk.LabelFrame(right, text="执行")
        lf_act.pack(fill=tk.X, pady=(0, 8))
        self._create_round_button(lf_act, "执行 OCR 并导出", self.run_ocr_export, accent=False, min_width=260).pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(right, text="建议：\n1. 先框选截图\n2. 采样对方和自己气泡\n3. 预览\n4. 导出", justify=tk.LEFT).pack(fill=tk.X)

    def _on_debug_toggle(self) -> None:
        enabled = bool(self.save_debug.get())
        self.save_debug_toggle.configure(
            text=f"调试输出：{'开启' if enabled else '关闭'}",
            bg="#2f80ff" if enabled else "#9eb5d2",
            activebackground="#5a9dff" if enabled else "#b4c7dd",
            selectcolor="#2f80ff" if enabled else "#9eb5d2",
        )

    def _reset_sample_state(self) -> None:
        self.other_hsv = None
        self.self_hsv = None
        hint = "未采样（将使用默认颜色，可能影响识别效果）"
        self.lbl_other.configure(text=hint, fg="#d4380d")
        self.lbl_self.configure(text=hint, fg="#d4380d")

    def _current_color_config(self) -> dict[str, Any] | None:
        if not self.other_hsv or not self.self_hsv:
            return None
        return {"other_hsv": list(self.other_hsv), "self_hsv": list(self.self_hsv), "h_tol": int(self.h_tol.get()), "s_tol": int(self.s_tol.get()), "v_tol": int(self.v_tol.get())}

    def choose_out_path(self) -> None:
        p = filedialog.asksaveasfilename(title="选择导出文件", defaultextension=".txt", filetypes=[("Text", "*.txt"), ("JSON", "*.json")])
        if p:
            self.out_path.set(p)

    def choose_debug_dir(self) -> None:
        p = filedialog.askdirectory(title="选择调试输出目录")
        if p:
            self.debug_dir.set(p)

    def capture_region(self) -> None:
        title = self.window_title.get().strip() or "微信"
        # 将 get_wechat_window 移到这里，且不应因找不到窗口而中断流程
        win = get_wechat_window(title)

        if win is None:
            # 如果没找到，只是给个提示，不直接 return
            self.status.set(f"提示：未找到标题包含「{title}」的窗口，请手动框选任意区域。")
        else:
            self.status.set(f"已激活窗口「{title}」，请开始框选聊天区域。")

        # 隐藏 GUI 准备截图
        self.root.withdraw()
        self.root.update_idletasks()
        time.sleep(0.3)  # 给窗口切换留出时间

        # 进入交互式框选
        region = select_region_interactive()

        if not region:
            self.root.deiconify()
            self.status.set("已取消框选。")
            return

        # 截图并恢复 GUI
        shot = pyautogui.screenshot(region=region)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        # 处理图片
        self.base_img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        self._reset_sample_state()
        self._show_image(self.base_img)
        self.status.set(f"截图成功：{region}。接下来请采样颜色或直接预览。")
    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", "图片读取失败")
            return
        self.base_img = img
        self._reset_sample_state()
        self._show_image(self.base_img)
        self.status.set(f"已加载：{path}")

    def start_pick(self, target: str) -> None:
        if self.base_img is None:
            messagebox.showinfo("提示", "请先截图或加载图片")
            return
        self.pick_target = target
        self.status.set("请在预览图上单击采样颜色")

    def on_canvas_click(self, e: tk.Event) -> None:
        if self.base_img is None or not self.pick_target:
            return
        ox, oy = self.view_offset
        x = int((e.x - ox) / self.scale_ratio)
        y = int((e.y - oy) / self.scale_ratio)
        h, w = self.base_img.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        bgr = self.base_img[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        hsv_tuple = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        if self.pick_target == "other":
            self.other_hsv = hsv_tuple
            self.lbl_other.configure(text=str(hsv_tuple), fg="#389e0d")
            self.status.set(f"已采样对方颜色：{hsv_tuple}")
        else:
            self.self_hsv = hsv_tuple
            self.lbl_self.configure(text=str(hsv_tuple), fg="#389e0d")
            self.status.set(f"已采样自己颜色：{hsv_tuple}")
        self.pick_target = None

    def preview_masks(self) -> None:
        if self.base_img is None:
            messagebox.showinfo("提示", "请先截图或加载图片")
            return
        color_config = self._current_color_config()
        white_mask, green_mask = _build_color_masks(self.base_img, color_config=color_config)
        h, w = self.base_img.shape[:2]
        bubbles = _extract_bubbles_from_mask(white_mask, "对方", w, h) + _extract_bubbles_from_mask(green_mask, "自己", w, h)
        vis = self.base_img.copy()
        for b in bubbles:
            x0, y0, x1, y1 = b["bbox"]
            color = (255, 0, 0) if b["sender"] == "对方" else (0, 180, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        self._show_image(vis)
        other_cnt = len([b for b in bubbles if b["sender"] == "对方"])
        mode_text = "默认阈值" if not color_config else "采样阈值"
        self.status.set(f"预览完成（{mode_text}）：对方{other_cnt}条，自己{len(bubbles) - other_cnt}条")

    def run_ocr_export(self) -> None:
        if self.base_img is None:
            messagebox.showinfo("提示", "请先截图或加载图片")
            return
        color_config = self._current_color_config()
        out = self.out_path.get().strip()
        if not out:
            messagebox.showinfo("提示", "请先设置导出文件路径")
            return
        dbg = self.debug_dir.get().strip() if self.save_debug.get() else None
        Path(out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if dbg:
            Path(dbg).mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, self.base_img)
            rows = _run_ocr_on_file(tmp_path, debug_dir=dbg, color_config=color_config)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        frames = [rows]
        meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "mode": "gui", "color_config": color_config if color_config else {"mode": "default_threshold"}}
        if out.lower().endswith(".json"):
            export_json(frames, out, meta)
        else:
            export_txt(frames, out, meta)
        mode_text = "默认阈值" if not color_config else "采样阈值"
        self.status.set(f"导出完成（{mode_text}）：{out}（共{len(rows)}条）")
        messagebox.showinfo("完成", f"导出成功\n路径: {out}\n有效消息: {len(rows)}\n颜色模式: {mode_text}")

    def _show_image(self, img_bgr: np.ndarray) -> None:
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        h, w = img_bgr.shape[:2]
        ratio = max(0.1, min(cw / w, ch / h))
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.preview_imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.canvas.create_image(ox, oy, image=self.preview_imgtk, anchor=tk.NW)
        self.scale_ratio = ratio
        self.view_offset = (ox, oy)
