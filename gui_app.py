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
    merge_scrolled_frames,
    scroll_and_collect,
    select_region_interactive,
)


class OcrTunerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("WeChat OCR 专家级调参工具")
        self.root.geometry("1500x950")
        self.root.minsize(1350, 850)

        # 定义全局统一背景色
        self.bg_color = "#f5f9ff"
        self.root.configure(bg=self.bg_color)

        self.ui_font_family = self._pick_font_family()
        self.root.option_add("*Font", (self.ui_font_family, 11))

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground="#1f2d3d")
        self.style.configure("TLabelframe", background=self.bg_color, foreground="#2b4b7c", bordercolor="#b9d4f7")
        self.style.configure("TLabelframe.Label", background=self.bg_color, foreground="#2b4b7c")
        self.style.configure("TButton", padding=7)

        # 核心数据状态
        self.base_img: np.ndarray | None = None
        self.preview_imgtk: ImageTk.PhotoImage | None = None
        self.scale_ratio = 1.0
        self.view_offset = (0, 0)
        self.pick_target: str | None = None
        self.selected_region: tuple[int, int, int, int] | None = None

        self.other_hsv: tuple[int, int, int] | None = None
        self.self_hsv: tuple[int, int, int] | None = None

        # 变量绑定
        self.h_tol = tk.IntVar(value=12)
        self.s_tol = tk.IntVar(value=60)
        self.v_tol = tk.IntVar(value=60)
        self.scroll_rounds = tk.IntVar(value=5)
        self.scroll_pause = tk.DoubleVar(value=0.8)
        self.save_debug = tk.BooleanVar(value=True)
        self.out_path = tk.StringVar(value=str(Path.cwd() / "wechat_export_gui.txt"))
        self.debug_dir = tk.StringVar(value=str(Path.cwd() / "debug_gui"))
        self.window_title = tk.StringVar(value="微信")
        self.status = tk.StringVar(value="准备就绪")

        # 交互增强变量
        self.show_mask_var = tk.BooleanVar(value=False)
        self.show_bubbles_var = tk.BooleanVar(value=False)
        self.highlighter: tk.Toplevel | None = None

        self._build_layout()

    def run(self) -> None:
        self.root.mainloop()

    def _pick_font_family(self) -> str:
        families = set(tkfont.families())
        for name in ("Source Han Sans CN", "Source Han Sans SC", "思源黑体", "Microsoft YaHei UI"):
            if name in families: return name
        return "TkDefaultFont"

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root)
        root_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        root_frame.columnconfigure(0, weight=4)
        root_frame.columnconfigure(1, weight=1)

        # --- 左侧区域 ---
        left_side = ttk.Frame(root_frame)
        left_side.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        toolbar = ttk.Frame(left_side)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        self._create_round_button(toolbar, "📷 框选截图", self.capture_region).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "📂 加载图片", self.load_image).pack(side=tk.LEFT, padx=4)
        self._create_round_button(toolbar, "🎯 采样对方颜色", lambda: self.start_pick("other")).pack(side=tk.LEFT,
                                                                                                    padx=4)
        self._create_round_button(toolbar, "🎯 采样自己颜色", lambda: self.start_pick("self")).pack(side=tk.LEFT, padx=4)

        # 画布容器
        canvas_container = ttk.Frame(left_side)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_container, bg="#2c3e50", highlightthickness=1, highlightbackground="#b9d4f7")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        ttk.Label(left_side, textvariable=self.status, foreground="#57606f",
                  font=(self.ui_font_family, 10, "italic")).pack(fill=tk.X, pady=(5, 0))

        # --- 右侧控制面板 ---
        right_side = ttk.Frame(root_frame)
        right_side.grid(row=0, column=1, sticky="nsew")

        # 1. 实时预览控制
        lf_preview = ttk.LabelFrame(right_side, text="🔍 实时视觉预览")
        lf_preview.pack(fill=tk.X, pady=(0, 10))

        tk.Checkbutton(lf_preview, text="显示颜色掩码 (Mask)", variable=self.show_mask_var,
                       command=self.refresh_preview, bg=self.bg_color).pack(anchor="w", padx=10, pady=2)
        tk.Checkbutton(lf_preview, text="显示识别气泡框 (Bubbles)", variable=self.show_bubbles_var,
                       command=self.refresh_preview, bg=self.bg_color).pack(anchor="w", padx=10, pady=2)
        ttk.Button(lf_preview, text="强制刷新视图", command=self.refresh_preview).pack(fill=tk.X, padx=10, pady=5)

        # 2. 颜色与容差
        lf_color = ttk.LabelFrame(right_side, text="🎨 采样容差调整")
        lf_color.pack(fill=tk.X, pady=(0, 10))

        f_other = ttk.Frame(lf_color)
        f_other.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(f_other, text="对方:").pack(side=tk.LEFT)
        self.lbl_other = tk.Label(f_other, text="[未采样]", fg="#d4380d", bg=self.bg_color)
        self.lbl_other.pack(side=tk.LEFT, padx=5)

        f_self = ttk.Frame(lf_color)
        f_self.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(f_self, text="自己:").pack(side=tk.LEFT)
        self.lbl_self = tk.Label(f_self, text="[未采样]", fg="#d4380d", bg=self.bg_color)
        self.lbl_self.pack(side=tk.LEFT, padx=5)

        self._add_slider(lf_color, "H 容差", self.h_tol, 2, 40)
        self._add_slider(lf_color, "S 容差", self.s_tol, 10, 150)
        self._add_slider(lf_color, "V 容差", self.v_tol, 10, 150)

        # 3. 滚动与输出
        lf_setup = ttk.LabelFrame(right_side, text="⚙️ 滚动与导出设置")
        lf_setup.pack(fill=tk.X, pady=(0, 10))

        self._add_entry(lf_setup, "窗口标题:", self.window_title)
        self._add_entry(lf_setup, "滚动屏数:", self.scroll_rounds, is_spin=True, _from=2, _to=100)
        self._add_entry(lf_setup, "滚动间隔:", self.scroll_pause, is_spin=True, _from=0.2, _to=5.0)

        ttk.Separator(lf_setup, orient="horizontal").pack(fill=tk.X, pady=8, padx=10)

        self._add_path_box(lf_setup, "导出文件:", self.out_path, self.choose_out_path)
        self._add_path_box(lf_setup, "调试目录:", self.debug_dir, self.choose_debug_dir)

        tk.Checkbutton(lf_setup, text="保存调试截图 (Debug Mode)", variable=self.save_debug, bg=self.bg_color).pack(
            anchor="w", padx=10)

        # 4. 执行按钮
        lf_run = ttk.LabelFrame(right_side, text="🚀 开始执行")
        lf_run.pack(fill=tk.X, pady=(0, 10))

        self._create_round_button(lf_run, "单页导出", self.run_ocr_export, min_width=280).pack(pady=5, padx=10)
        self._create_round_button(lf_run, "自动滚动导出", self.run_scroll_ocr_export, accent=True, min_width=280).pack(
            pady=5, padx=10)

    def _add_slider(self, parent, label, var, f, t):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text=label, width=8).pack(side=tk.LEFT)
        scale = ttk.Scale(frame, from_=f, to=t, variable=var, orient=tk.HORIZONTAL,
                          command=lambda _: self.refresh_preview())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(frame, textvariable=var, width=3).pack(side=tk.LEFT)

    def _add_entry(self, parent, label, var, is_spin=False, _from=0, _to=0):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(frame, text=label, width=10).pack(side=tk.LEFT)
        if is_spin:
            ttk.Spinbox(frame, from_=_from, to=_to, textvariable=var, width=10).pack(side=tk.LEFT)
        else:
            ttk.Entry(frame, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _add_path_box(self, parent, label, var, cmd):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(frame, text=label).pack(anchor="w")
        entry_f = ttk.Frame(frame)
        entry_f.pack(fill=tk.X)
        ttk.Entry(entry_f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(entry_f, text="...", width=3, command=cmd).pack(side=tk.LEFT, padx=2)

    def _create_round_button(self, parent, text: str, command, accent: bool = False, min_width: int = 140) -> tk.Canvas:
        bg = "#2f80ff" if accent else "#ffffff"
        fg = "#ffffff" if accent else "#2f80ff"
        border = "#2f80ff"
        font_obj = tkfont.Font(family=self.ui_font_family, size=11, weight="bold")
        btn_w = max(min_width, font_obj.measure(text) + 40)

        # 修复点：直接使用 self.bg_color 而不是尝试从 parent 读取
        canvas = tk.Canvas(parent, width=btn_w, height=42, bg=self.bg_color, highlightthickness=0, bd=0)

        def draw_shape(color):
            canvas.delete("btn")
            r = 12
            x1, y1 = btn_w - 2, 40
            points = [2 + r, 2, x1 - r, 2, x1, 2, x1, 2 + r, x1, y1 - r, x1, y1, x1 - r, y1, 2 + r, y1, 2, y1, 2,
                      y1 - r, 2, 2 + r, 2, 2]
            canvas.create_polygon(points, smooth=True, splinesteps=20, fill=color, outline=border, width=1.5,
                                  tags="btn")
            canvas.create_text(btn_w // 2, 21, text=text, fill=fg, font=font_obj, tags="txt")

        draw_shape(bg)
        canvas.bind("<Button-1>", lambda _: command())
        canvas.bind("<Enter>", lambda _: draw_shape("#1e69de" if accent else "#f0f7ff"))
        canvas.bind("<Leave>", lambda _: draw_shape(bg))
        canvas.configure(cursor="hand2")
        return canvas

    def refresh_preview(self, *args):
        if self.base_img is None: return
        vis = self.base_img.copy()
        color_config = self._current_color_config()

        if self.show_mask_var.get() and color_config:
            white_mask, green_mask = _build_color_masks(vis, color_config=color_config)
            mask_overlay = np.zeros_like(vis)
            mask_overlay[white_mask > 0] = [255, 200, 200]
            mask_overlay[green_mask > 0] = [200, 255, 200]
            vis = cv2.addWeighted(vis, 0.6, mask_overlay, 0.4, 0)

        if self.show_bubbles_var.get():
            cfg = color_config if color_config else None
            white_m, green_m = _build_color_masks(self.base_img, color_config=cfg)
            h, w = self.base_img.shape[:2]
            bubbles = _extract_bubbles_from_mask(white_m, "对方", w, h) + _extract_bubbles_from_mask(green_m, "自己", w,
                                                                                                     h)
            for b in bubbles:
                x0, y0, x1, y1 = b["bbox"]
                color = (255, 50, 50) if b["sender"] == "对方" else (50, 200, 50)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                cv2.putText(vis, b["sender"], (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self._show_image(vis)

    def on_canvas_click(self, e: tk.Event) -> None:
        if self.base_img is None or not self.pick_target: return
        ox, oy = self.view_offset
        x = int((e.x - ox) / self.scale_ratio)
        y = int((e.y - oy) / self.scale_ratio)
        h, w = self.base_img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            bgr = self.base_img[y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_t = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
            if self.pick_target == "other":
                self.other_hsv = hsv_t
                self.lbl_other.configure(text=f"HSV{hsv_t}", fg="#389e0d")
            else:
                self.self_hsv = hsv_t
                self.lbl_self.configure(text=f"HSV{hsv_t}", fg="#389e0d")
            self.pick_target = None
            self.status.set(f"采样成功: {hsv_t}")
            self.refresh_preview()

    def show_scroll_highlighter(self, region):
        if self.highlighter: self.highlighter.destroy()
        x, y, w, h = region
        self.highlighter = tk.Toplevel()
        self.highlighter.overrideredirect(True)
        self.highlighter.attributes("-topmost", True)
        self.highlighter.attributes("-transparentcolor", "#000001")
        self.highlighter.geometry(f"{w + 10}x{h + 10}+{x - 5}+{y - 5}")
        canvas = tk.Canvas(self.highlighter, width=w + 10, height=h + 10, bg="#000001", highlightthickness=0)
        canvas.pack()
        canvas.create_rectangle(5, 5, w + 5, h + 5, outline="red", width=4, dash=(10, 5))
        canvas.create_text(w // 2, h // 2, text="⚠️ 自动识别中，请勿操作鼠标!", fill="red",
                           font=(self.ui_font_family, 14, "bold"))
        self.root.update()

    def run_scroll_ocr_export(self) -> None:
        if self.selected_region is None:
            messagebox.showinfo("提示", "请先点击“框选截图”确定采集区域")
            return
        out = self._validate_out_path()
        if not out: return
        rounds = self.scroll_rounds.get()
        pause = self.scroll_pause.get()
        title = self.window_title.get().strip() or "微信"
        win = get_wechat_window(title)
        if win is None:
            messagebox.showerror("错误", f"未找到窗口: {title}")
            return
        msg = f"即将开始自动滚动识别。\n请停止操作鼠标！"
        if not messagebox.askokcancel("确认自动滚动", msg): return
        self.show_scroll_highlighter(self.selected_region)

        def update_progress(curr, total):
            self.status.set(f"🚀 进度: 第 {curr}/{total} 屏识别中...")
            self.root.update()

        try:
            color_cfg = self._current_color_config()
            dbg = self._prepare_debug_dir()
            frames = scroll_and_collect(win, rounds=rounds, pause=pause, region=self.selected_region,
                                        debug_dir=dbg, color_config=color_cfg, progress_callback=update_progress)
            merged = merge_scrolled_frames(frames)
            self._export_frames(frames, out, {"mode": "gui_scroll"})
            messagebox.showinfo("完成", f"滚动导出成功！\n采集: {len(frames)} 屏\n总计: {len(merged)} 条消息")
        finally:
            if self.highlighter: self.highlighter.destroy()
            self.highlighter = None

    def _show_image(self, img_bgr: np.ndarray) -> None:
        self.root.update_idletasks()
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        h, w = img_bgr.shape[:2]
        ratio = min(cw / w, ch / h) * 0.95
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.preview_imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.create_image(ox, oy, image=self.preview_imgtk, anchor=tk.NW)
        self.scale_ratio, self.view_offset = ratio, (ox, oy)

    def capture_region(self) -> None:
        title = self.window_title.get().strip() or "微信"
        win = get_wechat_window(title)
        self.root.withdraw()
        try:
            # Let desktop/window compositor settle after hiding GUI.
            time.sleep(0.25)
            region = select_region_interactive()
            if region:
                self.selected_region = region
                # Keep GUI hidden while taking screenshot to avoid overlay artifacts.
                time.sleep(0.12)
                shot = pyautogui.screenshot(region=region)
                self.base_img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
                self.refresh_preview()
        finally:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

    def load_image(self) -> None:
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.base_img = img
                self.selected_region = None
                self.refresh_preview()

    def start_pick(self, target: str) -> None:
        if self.base_img is None: return
        self.pick_target = target
        self.status.set(f"正在采样【{'对方' if target == 'other' else '自己'}】颜色，请点击预览图...")

    def run_ocr_export(self) -> None:
        if self.base_img is None: return
        out = self._validate_out_path()
        if not out: return
        color_cfg = self._current_color_config()
        dbg = self._prepare_debug_dir()
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp, self.base_img)
            rows = _run_ocr_on_file(tmp, debug_dir=dbg, color_config=color_cfg)
            self._export_frames([rows], out, {"mode": "single"})
            messagebox.showinfo("成功", f"单页导出完成。")
        finally:
            if os.path.exists(tmp): os.remove(tmp)

    def choose_out_path(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt")
        if p: self.out_path.set(p)

    def choose_debug_dir(self):
        p = filedialog.askdirectory()
        if p: self.debug_dir.set(p)

    def _validate_out_path(self):
        p = self.out_path.get().strip()
        if not p: return None
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        return p

    def _prepare_debug_dir(self):
        if not self.save_debug.get(): return None
        d = self.debug_dir.get().strip()
        if d: Path(d).mkdir(parents=True, exist_ok=True)
        return d

    def _current_color_config(self):
        if not self.other_hsv or not self.self_hsv: return None
        return {
            "other_hsv": list(self.other_hsv), "self_hsv": list(self.self_hsv),
            "h_tol": self.h_tol.get(), "s_tol": self.s_tol.get(), "v_tol": self.v_tol.get()
        }

    def _export_frames(self, frames, out, meta):
        if out.lower().endswith(".json"):
            export_json(frames, out, meta)
        else:
            export_txt(frames, out, meta)


if __name__ == "__main__":
    app = OcrTunerGUI()
    app.run()
