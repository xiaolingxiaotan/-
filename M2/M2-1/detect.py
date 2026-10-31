import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="红绿灯检测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="输入图片路径")
    source.add_argument("--video", help="输入视频路径")
    source.add_argument("--camera", type=int, help="摄像头索引，如 0")
    parser.add_argument("--calib", required=True, help="标定结果 YAML 路径")
    parser.add_argument("--show", action="store_true", help="显示结果窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image:
        source = f"图片 {args.image}"
    elif args.video:
        source = f"视频 {args.video}"
    else:
        source = f"摄像头 {args.camera}"
    print(f"输入: {source}")
    print(f"标定: {args.calib}")


if __name__ == "__main__":
    main()
