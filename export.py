import argparse
import os
import subprocess
from ultralytics import YOLO

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def export_onnx(model_path, imgsz=(640, 640), batch_size=1, output_dir="exports"):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_name = f"{model_name}_bs{batch_size}_imgsz{imgsz[0]}x{imgsz[1]}.onnx"
    onnx_path = os.path.join(output_dir, onnx_name)

    if os.path.exists(onnx_path):
        print(f"[ONNX] Found existing ONNX at {onnx_path}, skipping export.")
    else:
        print(f"[ONNX] Exporting ONNX from {model_path}...")
        model = YOLO(model_path)
        model.fuse()
        model.info(verbose=False)
        model.export(format='onnx', imgsz=imgsz, batch=batch_size)
        os.rename("model/yolov8n.onnx", onnx_path)
        print(f"[ONNX] Exported to {onnx_path}")

    return onnx_path

def export_trt(onnx_path, output_dir="exports", precision="fp16", workspace=8, end2end=False, v8=False, v10=False):
    trt_name = os.path.splitext(os.path.basename(onnx_path))[0] + f"_{precision}.engine"
    trt_path = os.path.join(output_dir, trt_name)

    if os.path.exists(trt_path):
        print(f"[TENSORRT] Found existing TensorRT engine at {trt_path}, skipping build.")
    else:
        print(f"[TENSORRT] Building TensorRT engine from {onnx_path}...")
        cmd = [
            "python3", "./tensorrt-for-yolo-series/export.py",
            "-o", onnx_path,
            "-e", trt_path,
            "-p", precision,
            "--workspace", str(workspace)
        ]
        if end2end:
            cmd.append("--end2end")
        if v8:
            cmd.append("--v8")
        if v10:
            cmd.append("--v10")

        subprocess.run(cmd, check=True)
        print(f"[TENSORRT] Exported to {trt_path}")

    return trt_path
  

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX/TensorRT")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], help="Input image size (h w)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for export")
    parser.add_argument("--format", choices=["onnx", "trt", "both"], default="both", help="Export format")
    parser.add_argument("--precision", type=str, default="fp16", help="TensorRT precision")
    parser.add_argument("--workspace", type=int, default=8, help="TensorRT workspace size")
    parser.add_argument("--output_dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--end2end", action="store_true", help="Enable end2end export for TensorRT")
    parser.add_argument("--v8", action="store_true", help="Specify YOLOv8 model")
    parser.add_argument("--v10", action="store_true", help="Specify YOLOv10 model")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    onnx_path = None
    if args.format in ["onnx", "both"]:
        onnx_path = export_onnx(args.weights, tuple(args.imgsz), args.batch, args.output_dir)

    if args.format in ["trt", "both"]:
        if onnx_path is None:
            raise ValueError("ONNX file required for TensorRT export. Use --format both or export ONNX first.")
        export_trt(onnx_path, args.output_dir, precision=args.precision, workspace=args.workspace, end2end=args.end2end, v8=args.v8, v10=args.v10)

if __name__ == "__main__":
    main()
