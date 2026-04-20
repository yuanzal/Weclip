from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from gui_app import OcrTunerGUI
from ocr_core import (
    export_json,
    export_txt,
    get_wechat_window,
    ocr_chat_region,
    scroll_and_collect,
    select_region_interactive,
)


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        OcrTunerGUI().run()
        return

    p = argparse.ArgumentParser(description="微信聊天区域截图 OCR 导出工具（无参启动 GUI，带参数启动 CLI）")
    p.add_argument("--gui", action="store_true", help="启动 GUI 调参工具")
    p.add_argument("--title", default="微信", help="窗口标题关键字")
    p.add_argument("--rounds", type=int, default=1, help="滚动采集次数；1=仅当前屏")
    p.add_argument("--pause", type=float, default=0.8, help="每屏滚动后等待秒数")
    p.add_argument("--out", default="", help="输出路径；支持 .json 或 .txt，默认按时间生成")
    p.add_argument("--keep-image", default="", help="保存调试截图到指定路径（用于调整坐标）")
    p.add_argument("--debug-dir", default="", help="调试输出目录：保存掩码、气泡框、裁剪图和 debug_log.json")
    p.add_argument(
        "--region",
        choices=("manual", "auto"),
        default="manual",
        help="识别区域：manual=启动后拖动框选（默认）；auto=按窗口布局推算",
    )
    args = p.parse_args(argv)

    if args.gui:
        OcrTunerGUI().run()
        return

    win = get_wechat_window(args.title)
    if not win:
        print(f"未找到标题包含“{args.title}”的窗口。")
        return

    user_region: tuple[int, int, int, int] | None = None
    if args.region == "manual":
        print("请在前台全屏遮罩上拖动鼠标，框选需要 OCR 的聊天区域。")
        user_region = select_region_interactive()
        if user_region is None:
            print("已取消，未框选区域。")
            return
        print(
            f"已选定区域: left={user_region[0]}, top={user_region[1]}, "
            f"w={user_region[2]}, h={user_region[3]}"
        )

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "window_title": args.title,
        "rounds_requested": args.rounds,
        "region_mode": args.region,
    }

    if args.rounds <= 1:
        rows = ocr_chat_region(
            win,
            image_path=args.keep_image or None,
            region=user_region,
            debug_dir=args.debug_dir or None,
        )
        frames = [rows]
        if args.keep_image:
            print(f"已保存调试截图: {args.keep_image}")
    else:
        frames = scroll_and_collect(
            win,
            args.rounds,
            args.pause,
            region=user_region,
            debug_dir=args.debug_dir or None,
        )

    out = args.out.strip()
    if not out:
        out = datetime.now().strftime("wechat_export_%Y%m%d_%H%M%S.txt")

    if out.lower().endswith(".json"):
        export_json(frames, out, meta)
    else:
        export_txt(frames, out, meta)

    total = sum(len(frame) for frame in frames)
    print(f"\n导出完成：共 {len(frames)} 屏，{total} 条有效聊天记录。")
    print(f"文件路径: {out}")


if __name__ == "__main__":
    main()
