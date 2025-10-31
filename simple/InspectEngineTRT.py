import tensorrt as trt

TRT_LOGGER = trt.Logger()

def inspect_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Number of bindings: {engine.num_bindings}")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            print(f"{'Input' if is_input else 'Output'}: {name}, shape={shape}, dtype={dtype}")

inspect_engine("best640lobb_myTRT.engine")
