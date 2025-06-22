import argparse
import cv2
import time
import os
import numpy as np
from infer import TensorRTv8
from detector import yoloDetecTRT  
import pycuda.driver as cuda


def load_images_from_dir(dir_path):
    images = []
    filenames = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(dir_path, fname)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(fname)
    return images, filenames

def process_images(detector, img_list, filenames, batch_size=1):
    total_time = 0
    frame_idx = 0

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        black_img = np.zeros((640, 640, 3), dtype=np.uint8)
        detector([black_img]*batch_size)

    for i in range(0, len(img_list), batch_size):
        batch_imgs = img_list[i:i + batch_size]
        batch_filenames = filenames[i:i + batch_size]

        # Padding the batch if it's not full
        if len(batch_imgs) < batch_size:
            pad_img = np.zeros((640, 640, 3), dtype=np.uint8)
            batch_imgs += [pad_img] * (batch_size - len(batch_imgs))

        start = time.time()
        results = detector(batch_imgs)
        end = time.time()

        latency = end - start
        total_time += latency
        frame_idx += 1

        avg_time = total_time / frame_idx
        fps = 1. / avg_time

        for img, res, name in zip(batch_imgs, results, batch_filenames):
            print(f"[{name}] avg_time: {avg_time:.4f} sec | FPS: {fps:.2f}")
            for det in res:
                x0, y0, x1, y1 = det['bbox']
                cls = det['class_id']
                score = det['score']
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, f"{cls} {score:.2f}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(f"{name}", img)
           
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector, video_path, batch_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    total_time= 0
    frame_idx = 0
    batch = []

    print("WarmUp")
    for i in range(50):        
        black_img= np.zeros([640,640,3],dtype=np.uint8)
        detector([black_img]*batch_size)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(batch)<batch_size:
            batch.append(frame)
            continue
        
        start = time.time()
        results = detector(batch)
        end = time.time()

        latency = end - start
        total_time += latency
        frame_idx += 1
        avg_time= total_time / (frame_idx*batch_size)
        fps = 1. / avg_time

        print("avg_time: {:.4f}".format(avg_time))
        print("FPS: {:.4f}".format(fps))
 

        res = results[0]
        for det in res:
            x0, y0, x1, y1 = det['bbox']
            cls = det['class_id']
            score = det['score']
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {score:.2f}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        batch.clear()
    
        # cv2.imshow("Video", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--use_dali', action='store_true', help='Use DALI preprocessing')
    parser.add_argument('--end2end', action='store_true', help='Use end-to-end TensorRT model for detection')
    parser.add_argument('--model_path', type=str, default="./model/yolov8n_640_batch1_end2end_fp16.engine")
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640])
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    try:
        engine = TensorRTv8(args.model_path, ctx=ctx)
        detector = yoloDetecTRT(input_size=tuple(args.input_size), batch_size=args.batch_size, engine=engine, det_conf=0.5, use_dali=args.use_dali, use_end2end=args.end2end)

        if args.image_dir:
            imgs, names = load_images_from_dir(args.image_dir)
            process_images(detector, imgs, names, args.batch_size)
        elif args.video:
            process_video(detector, args.video, batch_size=args.batch_size)
        else:
            print("Please provide --image_dir or --video")

    finally:
        ctx.pop()

if __name__ == "__main__":
    main()

# dali batch 1 : fps 268.42 latency 0.0038 (end2end) (per batch)
# opencv batch 1:  fps 205.2 latency 0.0049 (end2end) (per batch)
# dali batch 30 : fps 857.53  latency 0.0012(end2end) (per frame) latency:0.036 s fps: 27.8 (per batch)
# opencv batch 30: fps 302.88 latency 0.0033(end2end) (per frame) latency:0.099 s fps: 10.1 (per batch)


# dali batch 1 : fps 201.73 latency 0.005  (per batch)
# opencv batch 1:  fps 162.43 latency 0.0062  (per batch)
# dali batch 30 : fps 417.1  latency 0.0024 (per frame) latency:0.072 s fps: 13.9 (per batch)
# opencv batch 30: fps 225.21 latency 0.0044(per frame) latency:0.132 s fps: 7.5 (per batch)