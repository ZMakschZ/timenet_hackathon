"""Batch SCADA examples through the pinned OpenTSLM modules without changing their weights."""

from collections import defaultdict
from contextlib import contextmanager

import torch
from torch.nn import functional as F


def microbatches(rows, size, rng):
    """Bucket by window length and channel count so no sensor channels need to be padded."""
    buckets = defaultdict(list)
    for row in rows:
        series = row["example"].series
        shape = len(series), max(len(channel.values) for channel in series)
        buckets[shape].append(row)
    batches = []
    for pool in buckets.values():
        rng.shuffle(pool)
        batches.extend(pool[start : start + size] for start in range(0, len(pool), size))
    rng.shuffle(batches)
    return batches


@contextmanager
def _conditioned_batch(model, values, observed):
    batch, channels, width = values.shape
    flat_values = values.reshape(batch * channels, width)
    flat_observed = observed.reshape(batch * channels, width)
    patches = model.flamingo.vision_encoder(flat_values, flat_observed).unflatten(0, (batch, channels))
    patch_mask = observed.reshape(batch, channels, -1, model.patch_size).any(dim=-1)
    perceiver = model.flamingo.perceiver
    media = patches + perceiver.media_position[:channels]
    latents = perceiver.latents.expand(batch, channels, -1, -1)
    for attention, feedforward in zip(perceiver.attention_layers, perceiver.feedforward_layers, strict=True):
        latents = attention(media, latents, patch_mask) + latents
        latents = feedforward(latents) + latents
    latents = perceiver.norm(latents)
    layers = model.flamingo.lang_encoder.language_model.layers
    try:
        for layer in layers:
            layer.vis_x = latents
        yield
    finally:
        for layer in layers:
            layer.vis_x = layer.media_locations = None


def answer_losses(model, examples, questions):
    """Return summed answer-token losses and token counts for each example."""
    tokens, targets, values, observed = [], [], [], []
    for example, question in zip(examples, questions, strict=True):
        prompt, series, mask = model.prepare(example.series, question)
        answer = example.answer if prompt.endswith((" ", "\n")) else f" {example.answer}"
        target_ids = model.tokenizer(answer, add_special_tokens=False)["input_ids"] + [model.tokenizer.eos_token_id]
        tokens.append(model.tokenizer(prompt)["input_ids"] + target_ids)
        targets.append(target_ids)
        values.append(series)
        observed.append(mask)

    width = max(map(len, tokens))
    answer_width = max(map(len, targets))
    # Left padding right-aligns the target tokens without projecting the prompt through the vocabulary head.
    pad = model.tokenizer.pad_token_id
    ids = torch.tensor([[pad] * (width - len(row)) + row for row in tokens], device=model.device)
    mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in tokens], device=model.device)
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
    labels = torch.tensor([[-100] * (answer_width - len(row)) + row for row in targets], device=model.device)
    input_values = torch.stack(values).to(model.device)
    input_observed = torch.stack(observed).to(model.device)
    with _conditioned_batch(model, input_values, input_observed):
        logits = model.flamingo.lang_encoder(
            ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            logits_to_keep=answer_width + 1,
        ).logits
    losses = F.cross_entropy(logits[:, :-1].transpose(1, 2).float(), labels, reduction="none")
    return losses.sum(dim=1), tuple(map(len, targets))
