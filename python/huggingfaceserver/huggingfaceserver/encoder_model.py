# Copyright 2024 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pathlib
from typing import Any, Callable, Dict, Optional, Union

import torch
from accelerate import init_empty_weights
from kserve import Model
from kserve.logging import logger
from kserve.model import PredictorConfig
from kserve.protocol.infer_type import InferInput, InferRequest, InferResponse
from kserve.utils.utils import (
    from_np_dtype,
)
import asyncio
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
    Pipeline,
)

from .task import (
    MLTask,
    get_pipeline_for_task,
    is_generative_task,
    get_model_class_for_task,
    infer_task_from_model_architecture,
)


class PredictorProxyModel(PreTrainedModel):
    """
    This class acts like a Huggingface PreTrainedModel but for its forward pass
    it forwards the request to the predictor server for inference.
    """

    config: PretrainedConfig
    _modules = {}
    _parameters = {}
    _buffers = {}
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        config: PretrainedConfig,
        predict: Callable,
        model_name: str,
        input_names: Optional[str] = None,
    ):
        self.config = config
        self.predict = predict
        self.model_name = model_name
        self.input_names = input_names

    def __call__(self, **model_inputs):
        """
        Run inference. We will send an inference request to the predictor to do
        the actual work.
        """
        infer_inputs = []
        for key, input_tensor in model_inputs.items():
            # Send only specific inputs if they have been provided, otherwise send everything.
            if not self.input_names or key in self.input_names:
                infer_input = InferInput(
                    name=key,
                    datatype=from_np_dtype(input_tensor.numpy().dtype),
                    shape=list(input_tensor.shape),
                    data=input_tensor.numpy(),
                )
                infer_inputs.append(infer_input)
        infer_request = InferRequest(
            infer_inputs=infer_inputs, model_name=self.model_name
        )
        # Since predict is async it returns a coroutine. This needs to be run in an event loop.
        res = asyncio.run_coroutine_threadsafe(self.predict(infer_request), self.loop)
        res = res.result()
        return {out.name: torch.Tensor(out.data).view(out.shape) for out in res.outputs}


class HuggingfaceEncoderModel(Model):  # pylint:disable=c-extension-no-member
    task: MLTask
    model_config: PretrainedConfig
    model_id_or_path: Union[pathlib.Path, str]
    do_lower_case: bool
    add_special_tokens: bool
    max_length: Optional[int]
    tensor_input_names: Optional[str]
    return_token_type_ids: Optional[bool]
    model_revision: Optional[str]
    tokenizer_revision: Optional[str]
    trust_remote_code: bool
    ready: bool = False
    pipeline: Pipeline
    max_threadpool_workers: Optional[int] = None
    _tokenizer: PreTrainedTokenizerBase
    _model: Optional[PreTrainedModel] = None
    _device: torch.device
    _executor: ThreadPoolExecutor

    def __init__(
        self,
        model_name: str,
        model_id_or_path: Union[pathlib.Path, str],
        model_config: Optional[PretrainedConfig] = None,
        task: Optional[MLTask] = None,
        do_lower_case: bool = False,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        tensor_input_names: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        trust_remote_code: bool = False,
        max_threadpool_workers: Optional[int] = None,
        predictor_config: Optional[PredictorConfig] = None,
    ):
        super().__init__(model_name, predictor_config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id_or_path = model_id_or_path
        self.do_lower_case = do_lower_case
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.dtype = dtype
        self.tensor_input_names = tensor_input_names
        self.return_token_type_ids = return_token_type_ids
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.trust_remote_code = trust_remote_code
        self.max_threadpool_workers = max_threadpool_workers

        if model_config:
            self.model_config = model_config
        else:
            self.model_config = AutoConfig.from_pretrained(self.model_id_or_path)

        if task:
            self.task = task
            try:
                inferred_task = infer_task_from_model_architecture(self.model_config)
            except ValueError:
                inferred_task = None
            if inferred_task is not None and inferred_task != task:
                logger.warn(
                    f"Inferred task is '{inferred_task.name}' but"
                    f" task is explicitly set to '{self.task.name}'"
                )
        else:
            self.task = infer_task_from_model_architecture(self.model_config)

        if is_generative_task(self.task):
            raise RuntimeError(
                f"Encoder model does not support generative task: {self.task.name}"
            )

    def load(self) -> bool:
        model_id_or_path = self.model_id_or_path
        if self.max_length is None:
            self.max_length = self.model_config.max_length

        # device_map = "auto" enables model parallelism but all model architcture dont support it.
        # For pre-check we initialize the model class without weights to check the `_no_split_modules`
        # device_map = "auto" for models that support this else set to either cuda/cpu
        with init_empty_weights():
            self._model = AutoModel.from_config(self.model_config)

        device_map = self._device

        if self._model._no_split_modules:
            device_map = "auto"

        tokenizer_kwargs = {}
        model_kwargs = {}

        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            tokenizer_kwargs["trust_remote_code"] = True

        model_kwargs["torch_dtype"] = self.dtype

        # load huggingface tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_id_or_path),
            revision=self.tokenizer_revision,
            do_lower_case=self.do_lower_case,
            **tokenizer_kwargs,
        )
        logger.info("Successfully loaded tokenizer")

        # load huggingface model using from_pretrained for inference mode
        if not self.predictor_host:
            # If loading the model locally we want a single threadpool worker as we only
            # want to run a single forward pass at a time.
            max_threadpool_workers = self.max_threadpool_workers or 1
            model_cls = get_model_class_for_task(self.task)
            self._model = model_cls.from_pretrained(
                model_id_or_path,
                revision=self.model_revision,
                device_map=device_map,
                **model_kwargs,
            )
            self._model.eval()
            self._model.to(self._device)
            if not self._tokenizer.pad_token:
                pad_token_str = "[PAD]"
                logger.warning(
                    f"Tokenizer does not have a padding token defined. Adding fall back pad token `{pad_token_str}`"
                )
                # Add fallback pad token [PAD]
                self._tokenizer.add_special_tokens({"pad_token": pad_token_str})
                # When adding new tokens to the vocabulary, we should make sure to also resize the token embedding
                # matrix of the model so that its embedding matrix matches the tokenizer.
                self._model.resize_token_embeddings(len(self._tokenizer))
            logger.info(
                f"Successfully loaded huggingface model from path {model_id_or_path}"
            )
        else:
            # If the model is remote use the default number of threadpool workers as it
            # is configured to parallelize IO.
            max_threadpool_workers = self.max_threadpool_workers
            self._model = PredictorProxyModel(
                self.model_config,
                model_name=self.name,
                predict=super().predict,
                input_names=self.tensor_input_names,
            )
        self.pipeline = get_pipeline_for_task(
            self.task,
            self._model,
            self._tokenizer,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_threadpool_workers)
        self.ready = True
        return self.ready

    async def predict(
        self,
        payload: Dict,
        context: Dict[str, Any],
    ) -> Union[Tensor, InferResponse]:
        if isinstance(self._model, PredictorProxyModel) and not hasattr(
            self._model, "loop"
        ):
            self._model.loop = asyncio.get_running_loop()
        # Run the inference in a thread-pool executor. Since the call to `pipeline` is
        # blocking this ensures we don't block the event loop.
        output = await asyncio.get_running_loop().run_in_executor(
            self._executor, partial(self.pipeline, **payload)
        )
        if self.task == MLTask.token_classification:
            for s in output:
                for entity in s:
                    entity["score"] = entity["score"].item()

        return output
