import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from utils import with_cuda_context
from .Base_engine import BaseTensorRTInfer

class TensorRTv8(BaseTensorRTInfer):
    """
    TensorRTv8 模型封裝類
    Inputs: (1, 3, 640, 640)
    Outputs:
    - num (1, 1)
    - boxes (1, 100, 4)
    - scores (1, 100)
    - classes (1, 100)
    """
    def __init__(self,model_path, ctx):
        super().__init__(model_path=model_path, ctx=ctx)
     
        self.output_ptrs = [] # 儲存推理後資料（GPU）
        self.output_hosts = [] # 儲存推理後的資料（CPU）
        self.bindings = [] # 所有I/O 接口 （input + output）
   
        for idx, binding in enumerate(self.engine):
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(idx):
                self.input_binding_idx = idx 
                self.input_host = host_mem
                self.input_device = device_mem
            else:
                self.output_ptrs.append(device_mem)
                self.output_hosts.append(host_mem)
   
    def get_input_nodes_name_shape(self):
        input_nodes = []
        for idx, binding in enumerate(self.engine):
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                shape = self.engine.get_binding_shape(name)
                input_nodes.append((name, shape))
        return input_nodes
    
    @with_cuda_context
    def infer_numpy(self, input_array: np.ndarray):
        assert isinstance(input_array, np.ndarray)
        flat = np.ascontiguousarray(input_array.ravel(), dtype=self.input_host.dtype)
        self.input_host[:] = flat

        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)

        self.bindings[self.input_binding_idx] = int(self.input_device)
        for i, device in enumerate(self.output_ptrs):
            self.bindings[i+1] = int(device)  # output start idx
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for host, device in zip(self.output_hosts, self.output_ptrs):
            cuda.memcpy_dtoh_async(host, device, self.stream)

        self.stream.synchronize()
        return [np.copy(host) for host in self.output_hosts]
        
    @with_cuda_context
    def infer(self, input_ptrs):
        """
            input_ptrs: list of gpu ptrs 
            return: List of output numpy arrays (each shape matches output tensor)
        """
        try:
            # input 
            self.bindings[self.input_binding_idx] = int(input_ptrs[0])  
            
            # 設定output 
            for i in range(1, self.engine.num_bindings):
                self.bindings[i] = self.output_ptrs[i - 1]

            # 推理
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # 拷貝回host
            for host, device in zip(self.output_hosts, self.output_ptrs):
                cuda.memcpy_dtoh_async(host, device, self.stream)
            self.stream.synchronize()
            
            return [np.copy(host) for host in self.output_hosts]
        
        except Exception as e:
            raise RuntimeError(f"[TensorRTV8] 推論失敗: {e}")


def main():
    from pipeline import build_preproc,  help_preproc
    import cv2
    from trt_infer_tool import postprocess_batch
    cuda.init()
    device = cuda.Device(0)
    try:
        ctx = device.make_context() 
        model_path = '/workspace/dali_to_git/model/yolov8n_640_batch1_end2end_fp16.engine'
        engine = TensorRTv8(model_path, ctx)
        img_path = "/workspace/dali_to_git/test.jpg"
        img = cv2.imread(img_path)
        #pre = LetterBoxPreproc(batch=1)
        help_preproc("opencv")
        pre = build_preproc("opencv")

        img_out, r = pre([img]) # , input_size=(640, 640)
        print(r)
        img_debug = (np.transpose(img_out[0], (1, 2, 0)) * 255).astype(np.uint8)
        cv2.imwrite("opencv_preproc_debug.jpg", img_debug)

        # batch_tensors = img_out.as_tensor()
        # np_img = img_out.as_cpu().as_array()[0]
        # np_img = (np.transpose(np_img, (1, 2, 0)) * 255).astype(np.uint8)
        # cv2.imwrite("dali_preproc_debug.jpg", np_img)
        # input_ptr = [batch_tensors.data_ptr()]
        data = engine.infer_numpy(img_out)
        #data = engine.infer(input_ptr)
        dets = postprocess_batch(data, r, batch_size=1,  end2end=True, score_thresh=0.5)
        print(dets)
    finally:
        ctx.pop()
    

if __name__=='__main__':
    main()