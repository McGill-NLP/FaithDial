import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

HF_CLASSIFICATION_HEADS = {
    "electra": {"head": ElectraClassificationHead},
    "roberta": {"head": RobertaClassificationHead},
    "bert": {"head": (lambda config: nn.Linear(config.hidden_size, config.num_labels))},
}


def _reinit_classifier_head(config: PretrainedConfig, model: PreTrainedModel):
    for module in model.classifier.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def reinitialize(model_name: str, config: PretrainedConfig, model: PreTrainedModel):
    if model_name not in HF_CLASSIFICATION_HEADS:
        raise NotImplementedError(f"Unknown model: {model_name}")

    meta = HF_CLASSIFICATION_HEADS[model_name]
    model.classifier = meta["head"](config)
    _reinit_classifier_head(config, model)
