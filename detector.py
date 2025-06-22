import pycuda.driver as cuda
from pipeline import build_preproc, help_preproc
from trt_infer_tool import postprocess_batch
from typing import Tuple



class yoloDetecTRT:
    def __init__(self, 
                input_size: Tuple[int, int], 
                batch_size: int, 
                engine, 
                det_conf:float, 
                use_end2end: bool = True,
                normalize: bool = True,
                use_dali: bool = False,
                num_classes: int = 80,
                device: int = 0,
                num_threads: int = 2
            ):
        """
        batchsz: 一次推理幾張圖
        engine: 過tensorRT初始化的trt模型
        det_conf: 檢測器閥值
        normaleize: 前處理啟用正歸化
        use_dali: 前處理使用Dali yolo pipeline 否則用opencv
        num_classes: 總類別數
        device: gpu 編號
        num_thread : dali用開多少條執行緒處理
        """
        self.engine = engine

        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device
        self.num_threads = num_threads
        self.num_classes = num_classes
        self.conf = det_conf

        self.end2end = use_end2end
        self.normalize = normalize
        self.use_dali = use_dali

        if self.use_dali:
            help_preproc("dali")
            self.preproc = build_preproc("dali", input_size=input_size, batch_size=batch_size, device_id=device, num_threads = num_threads, normalize=normalize)
        else:
            help_preproc("opencv")
            self.preproc = build_preproc("opencv", batch_size=batch_size, input_size = input_size, normalize=normalize, to_rgb=True)
        
        
    def _infer(self, batch_datas):
        """
        實現dali/opencv 前處理 TensorRT推理
        """
        if self.use_dali:
            batch_tensors = batch_datas.as_tensor()
            input_ptr = [batch_tensors.data_ptr()]
            dets = self.engine.infer(input_ptr)
        else:
            dets = self.engine.infer_numpy(batch_datas)
        return dets
    
    def __call__(self, img_list):
        """
        實現前處理--> 推理--> 後處理
        """
        batch_datas, batch_ratios = self.preproc.run(img_list)
        dets = self._infer(batch_datas=batch_datas)
        output = postprocess_batch(dets, batch_ratios, batch_size=self.batch_size, num_classes=self.num_classes, score_thresh=self.conf, end2end=self.end2end)
        return output


if __name__ == "__main__":
    from infer import TensorRTv8
    import cv2
    img_path = "test.jpg"
    model_path = "./model/yolov8n_640_batch1_end2end_fp16.engine"
    cuda.init()
    device = cuda.Device(0)
    try:
        ctx = device.make_context() # 自定cuda context
        engine = TensorRTv8(model_path, ctx=ctx)
        detector = yoloDetecTRT(input_size=(640, 640), batch_size=1, engine=engine, det_conf=0.5, use_dali=True, use_end2end=True)

        img = cv2.imread(img_path)
        output = detector([img])
        print(f'推理結果：{output}')

    finally:
        ctx.pop() # 釋放資源






    


    