# Copyright (c) 2024 The InvokeAI Development Team
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnxruntime import InferenceSession, SessionOptions, get_available_providers

ONNX_WEIGHTS_NAME = "model.onnx"


# NOTE FROM LS: This was copied from Stalker's original implementation.
# I have not yet gone through and fixed all the type hints
class IAIOnnxRuntimeModel(torch.nn.Module):
    class _tensor_access:
        def __init__(self, model):  # type: ignore
            self.model = model
            self.indexes = {}
            for idx, obj in enumerate(self.model.proto.graph.initializer):
                self.indexes[obj.name] = idx

        def __getitem__(self, key: str):  # type: ignore
            value = self.model.proto.graph.initializer[self.indexes[key]]
            return numpy_helper.to_array(value)

        def __setitem__(self, key: str, value: np.ndarray):  # type: ignore
            new_node = numpy_helper.from_array(value)
            # set_external_data(new_node, location="in-memory-location")
            new_node.name = key
            # new_node.ClearField("raw_data")
            del self.model.proto.graph.initializer[self.indexes[key]]
            self.model.proto.graph.initializer.insert(self.indexes[key], new_node)
            # self.model.data[key] = OrtValue.ortvalue_from_numpy(value)

        # __delitem__

        def __contains__(self, key: str) -> bool:
            return self.indexes[key] in self.model.proto.graph.initializer

        def items(self) -> List[Tuple[str, Any]]:  # fixme
            raise NotImplementedError("tensor.items")
            # return [(obj.name, obj) for obj in self.raw_proto]

        def keys(self) -> List[str]:
            return list(self.indexes.keys())

        def values(self) -> List[Any]:  # fixme
            raise NotImplementedError("tensor.values")
            # return [obj for obj in self.raw_proto]

        def size(self) -> int:
            bytesSum = 0
            for node in self.model.proto.graph.initializer:
                bytesSum += sys.getsizeof(node.raw_data)
            return bytesSum

    class _access_helper:
        def __init__(self, raw_proto):  # type: ignore
            self.indexes = {}
            self.raw_proto = raw_proto
            for idx, obj in enumerate(raw_proto):
                self.indexes[obj.name] = idx

        def __getitem__(self, key: str):  # type: ignore
            return self.raw_proto[self.indexes[key]]

        def __setitem__(self, key: str, value):  # type: ignore
            index = self.indexes[key]
            del self.raw_proto[index]
            self.raw_proto.insert(index, value)

        # __delitem__

        def __contains__(self, key: str) -> bool:
            return key in self.indexes

        def items(self) -> List[Tuple[str, Any]]:
            return [(obj.name, obj) for obj in self.raw_proto]

        def keys(self) -> List[str]:
            return list(self.indexes.keys())

        def values(self) -> List[Any]:  # fixme
            return list(self.raw_proto)

    def __init__(self, model_path: str, provider: Optional[str]):
        self.path = model_path
        self.session = None
        self.provider = provider
        """
        self.data_path = self.path + "_data"
        if not os.path.exists(self.data_path):
            print(f"Moving model tensors to separate file: {self.data_path}")
            tmp_proto = onnx.load(model_path, load_external_data=True)
            onnx.save_model(tmp_proto, self.path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(self.data_path), size_threshold=1024, convert_attribute=False)
            del tmp_proto
            gc.collect()

        self.proto = onnx.load(model_path, load_external_data=False)
        """
        super().__init__()
        self.proto = onnx.load(model_path, load_external_data=True)
        # self.data = dict()
        # for tensor in self.proto.graph.initializer:
        #     name = tensor.name

        #     if tensor.HasField("raw_data"):
        #         npt = numpy_helper.to_array(tensor)
        #         orv = OrtValue.ortvalue_from_numpy(npt)
        #         # self.data[name] = orv
        #         # set_external_data(tensor, location="in-memory-location")
        #         tensor.name = name
        #         # tensor.ClearField("raw_data")

        self.nodes = self._access_helper(self.proto.graph.node)  # type: ignore
        # self.initializers = self._access_helper(self.proto.graph.initializer)
        # print(self.proto.graph.input)
        # print(self.proto.graph.initializer)

        self.tensors = self._tensor_access(self)  # type: ignore

    # TODO: integrate with model manager/cache
    def create_session(self, height=None, width=None):
        if self.session is None or self.session_width != width or self.session_height != height:
            # onnx.save(self.proto, "tmp.onnx")
            # onnx.save_model(self.proto, "tmp.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="tmp.onnx_data", size_threshold=1024, convert_attribute=False)
            # TODO: something to be able to get weight when they already moved outside of model proto
            # (trimmed_model, external_data) = buffer_external_data_tensors(self.proto)
            sess = SessionOptions()
            # self._external_data.update(**external_data)
            # sess.add_external_initializers(list(self.data.keys()), list(self.data.values()))
            # sess.enable_profiling = True

            # sess.intra_op_num_threads = 1
            # sess.inter_op_num_threads = 1
            # sess.execution_mode = ExecutionMode.ORT_SEQUENTIAL
            # sess.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            # sess.enable_cpu_mem_arena = True
            # sess.enable_mem_pattern = True
            # sess.add_session_config_entry("session.intra_op.use_xnnpack_threadpool", "1") ########### It's the key code
            self.session_height = height
            self.session_width = width
            if height and width:
                sess.add_free_dimension_override_by_name("unet_sample_batch", 2)
                sess.add_free_dimension_override_by_name("unet_sample_channels", 4)
                sess.add_free_dimension_override_by_name("unet_hidden_batch", 2)
                sess.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
                sess.add_free_dimension_override_by_name("unet_sample_height", self.session_height)
                sess.add_free_dimension_override_by_name("unet_sample_width", self.session_width)
                sess.add_free_dimension_override_by_name("unet_time_batch", 1)
            providers = []
            if self.provider:
                providers.append(self.provider)
            else:
                providers = get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")
            try:
                self.session = InferenceSession(self.proto.SerializeToString(), providers=providers, sess_options=sess)
            except Exception as e:
                raise e
            # self.session = InferenceSession("tmp.onnx", providers=[self.provider], sess_options=self.sess_options)
            # self.io_binding = self.session.io_binding()

    def release_session(self):
        self.session = None
        import gc

        gc.collect()
        return

    def __call__(self, **kwargs):
        if self.session is None:
            raise Exception("You should call create_session before running model")

        inputs = {k: np.array(v) for k, v in kwargs.items()}
        # output_names = self.session.get_outputs()
        # for k in inputs:
        #     self.io_binding.bind_cpu_input(k, inputs[k])
        # for name in output_names:
        #     self.io_binding.bind_output(name.name)
        # self.session.run_with_iobinding(self.io_binding, None)
        # return self.io_binding.copy_outputs_to_cpu()
        return self.session.run(None, inputs)

    # compatability with diffusers load code
    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        subfolder: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
        provider: Optional[str] = None,
        sess_options: Optional["SessionOptions"] = None,
        **kwargs: Any,
    ) -> Any:  # fixme
        file_name = file_name or ONNX_WEIGHTS_NAME

        if os.path.isdir(model_id):
            model_path = model_id
            if subfolder is not None:
                model_path = os.path.join(model_path, subfolder)
            model_path = os.path.join(model_path, file_name)

        else:
            model_path = model_id

        # load model from local directory
        if not os.path.isfile(model_path):
            raise Exception(f"Model not found: {model_path}")

        # TODO: session options
        return cls(str(model_path), provider=provider)
