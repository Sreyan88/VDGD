import torch
from typing import TYPE_CHECKING
from peft import PeftModel, TaskType, LoraConfig, get_peft_model

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    

def init_adapter(
    model: "PreTrainedModel",
    adapter_name_or_path: str,
    is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    adapter_to_resume = None

    if adapter_name_or_path is not None:
        adapter_to_merge = adapter_name_or_path
        print(adapter_to_merge)
        model = PeftModel.from_pretrained(model, adapter_to_merge)
        model = model.merge_and_unload()

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    if adapter_name_or_path is not None:
        print("Loaded adapter(s): {}".format((adapter_name_or_path)))

    return model