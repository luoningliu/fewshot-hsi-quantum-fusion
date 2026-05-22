from __future__ import annotations


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

