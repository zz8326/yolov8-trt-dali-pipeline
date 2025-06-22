import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class BaseTensorRTInfer:
    def __init__(self, model_path, ctx):
            self.model_path = model_path
            self.ctx = ctx
            self.logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(self.logger, '')
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.load_engine()
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

    def load_engine(self):
        with open(self.model_path, 'rb') as f:
            serialized_engine = f.read()
        return self.runtime.deserialize_cuda_engine(serialized_engine)

    def infer(self, input_ptrs):
        raise NotImplementedError("Subclasses must implement this!")

    def get_input_nodes_name_shape(self):
        raise NotImplementedError("Subclasses must implement this!")
        