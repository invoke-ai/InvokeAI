from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    InputField,
)
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice

AUGMENT_PROMPT_INSTRUCTION = """Your task is to translate a short image caption and a style caption to a more detailed caption for the same image. The detailed caption should adhere to the following:
- be 1 sentence long
- use descriptive language that relates to the subject of interest
- it may add new details, but shouldn't change the subject of the original caption
Here are some examples:
Original caption: "A cat on a table"
Detailed caption: "A fluffy cat with a curious expression, sitting on a wooden table next to a vase of flowers."
Original caption: "medieval armor"
Detailed caption: "The gleaming suit of medieval armor stands proudly in the museum, its intricate engravings telling tales of long-forgotten battles and chivalry."
Original caption: "A panda bear as a mad scientist"
Detailed caption: "Clad in a tiny lab coat and goggles, the panda bear feverishly mixes colorful potions, embodying the eccentricity of a mad scientist in its whimsical laboratory."
Here is the prompt to translate:
Original caption: "{}"
Detailed caption:"""


@invocation("promp_augment", title="Prompt Augmentation", tags=["prompt"], category="conditioning", version="1.0.0")
class PrompAugmentationInvocation(BaseInvocation):
    """Use an LLM to augment a text prompt."""

    prompt: str = InputField(description="The text prompt to augment.")

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> StringOutput:
        # TODO(ryand): Address the following situations in the input prompt:
        # - Prompt contains a TI embeddings.
        # - Prompt contains .and() compel syntax. (Is ther any other compel syntax we need to handle?)
        # - Prompt contains quotation marks that could cause confusion when embedded in an LLM instruct prompt.

        # Load the model and tokenizer.
        model_source = "microsoft/Phi-3-mini-4k-instruct"

        def model_loader(model_path: Path):
            return AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=TorchDevice.choose_torch_dtype(), local_files_only=True
            )

        def tokenizer_loader(model_path: Path):
            return AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        with (
            context.models.load_remote_model(source=model_source, loader=model_loader) as model,
            context.models.load_remote_model(source=model_source, loader=tokenizer_loader) as tokenizer,
        ):
            # Tokenize the input prompt.
            augmented_prompt = self._run_instruct_model(model, tokenizer, self.prompt)

        return StringOutput(value=augmented_prompt)

    def _run_instruct_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": AUGMENT_PROMPT_INSTRUCTION.format(prompt),
            }
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.9,
            do_sample=True,
        )
        text = tokenizer.batch_decode(outputs)[0]
        assert isinstance(text, str)

        output = text.split("<|assistant|>")[-1].strip()
        output = output.split("<|end|>")[0].strip()

        return output
