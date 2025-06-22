import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from .Base_engine import BaseTensorRTInfer
from utils import with_cuda_context


class TensorRTv10(BaseTensorRTInfer):
    """
    TensorRT v10 類
    - 只接收GPU指標輸入
    - input/output 記憶體分配
    - 推理函數
    engine 結構：
    [0] Name:images, shape:(1, 3, 640, 640), Dtype:DataType.FLOAT
    [1] Name:num, shape:(1, 1), Dtype:DataType.INT32
    [2] Name:boxes, shape:(1, 100, 4), Dtype:DataType.FLOAT
    [3] Name:scores, shape:(1, 100), Dtype:DataType.FLOAT
    [4] Name:classes, shape:(1, 100), Dtype:DataType.INT32

    """
    def __init__(self, model_path, ctx):
        super().__init__(model_path=model_path, ctx=ctx)
      
        self.inputs = []  # 存放input 訊息[{host_mem, device_mem, shape, name}]
        self.outputs = [] # 存放 output 訊息[{host_mem, device_mem, shape, name}]

        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding) 
            tensor_shape = tuple(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            h_nbytes = cuda.pagelocked_empty(trt.volume(tensor_shape), dtype)
            d_nbytes = cuda.mem_alloc(h_nbytes.nbytes)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, tensor_shape)
                self.inputs.append({'host': h_nbytes, 'device': d_nbytes, 'shape': tensor_shape,'name':tensor_name})
            else:
                self.outputs.append({
                    'host': h_nbytes,
                    'device': d_nbytes,
                    'shape': tensor_shape,
                    'name': tensor_name
                })


    def get_input_nodes_name_shape(self):
        input_nodes= []
        for binding in range(self.engine.num_io_tensors):
            node_name = self.engine.get_tensor_name(binding) 
            node_shape = tuple(self.engine.get_tensor_shape(node_name))
            
            if self.engine.get_tensor_mode(node_name) == trt.TensorIOMode.INPUT:
                input_nodes.append((node_name, node_shape))
                       
        return input_nodes
    
    def infer_numpy(self, input_array: np.ndarray):
        assert isinstance(input_array, np.ndarray), "input must be np.ndarray"
        ptrs = []
        shapes = []
        for info in self.inputs:
            shapes.append(input_array.shape)
            dptr = info['device']
            cuda.memcpy_htod_async(dptr, input_array, self.stream)
            ptrs.append(int(dptr))
        self.stream.synchronize()
        return self.infer(ptrs, shapes)
    
    @with_cuda_context
    def infer(self, input_ptrs, input_shapes):
        """
        Args:
        input_ptrs (List[int]): GPU pointer(s) to image(s)
        input_shapes (List[Tuple[int]]): Input tensor shapes (e.g., (1, 3, 640, 640))

        Returns:
            List[np.ndarray]: Output tensors copied back to CPU
        """
        try:
            for i, (ptr, shape) in enumerate(zip(input_ptrs, input_shapes)):
                info = self.inputs[i]
                self.context.set_input_shape(info['name'], shape)
                self.context.set_tensor_address(info['name'], int(ptr))
            
            # output
            for i,out in enumerate(self.outputs):
                self.context.set_tensor_address(out['name'], out['device'])

            # 推理
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # 拷貝回cpu
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()

            return [np.copy(out['host']) for out in self.outputs]

        except Exception as e:
            raise RuntimeError(f"[TensorRTV10] 推論失敗: {e}")  