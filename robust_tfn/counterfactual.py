import torch
import torch.nn.functional as F


def physical_anchor_weights(
    labels,
    rpms,
    centers,
    fault_ratios,
    sample_rate,
    bandwidth=0.006,
    relevant_fraction=0.25,
):
    """Return differentiable channel relevance weights for theoretical harmonics."""
    batch_weights = []
    for label, rpm in zip(labels.tolist(), rpms.tolist()):
        ratio = fault_ratios.get(int(label))
        if ratio is None:
            batch_weights.append(torch.zeros_like(centers))
            continue
        fundamental = ratio * rpm / 60.0 / sample_rate
        harmonics = torch.arange(1, 129, device=centers.device) * fundamental
        harmonics = harmonics[(harmonics >= 0.02) & (harmonics <= 0.48)]
        if harmonics.numel() == 0:
            harmonics = torch.tensor(
                [min(max(fundamental, 1e-4), 0.48)],
                device=centers.device,
                dtype=centers.dtype,
            )
        distances = torch.abs(centers[:, None] - harmonics[None, :])
        weights = torch.exp(-0.5 * (distances.min(dim=1).values / bandwidth) ** 2)
        count = max(int(round(len(centers) * relevant_fraction)), 1)
        selected = torch.topk(weights, k=count).indices
        sparse_weights = torch.zeros_like(weights)
        sparse_weights[selected] = weights[selected]
        sparse_weights = sparse_weights / sparse_weights.max().clamp_min(1e-8)
        batch_weights.append(sparse_weights)
    return torch.stack(batch_weights)


def control_weights(anchor_weights):
    shift = max(anchor_weights.shape[1] // 3, 1)
    return torch.roll(anchor_weights, shifts=shift, dims=1)


def counterfactual_ranking_loss(
    model,
    inputs,
    labels,
    rpms,
    margin=0.10,
    bandwidth=0.006,
):
    original = model(inputs, return_details=True)
    anchor = physical_anchor_weights(
        labels,
        rpms,
        model.center_frequencies,
        model.fault_ratios,
        model.sample_rate,
        bandwidth=bandwidth,
    )
    control = control_weights(anchor)
    related_logits = model(inputs, channel_mask=1.0 - anchor)
    control_logits = model(inputs, channel_mask=1.0 - control)

    probabilities = F.softmax(original["logits"], dim=1)
    related_probabilities = F.softmax(related_logits, dim=1)
    control_probabilities = F.softmax(control_logits, dim=1)
    indices = torch.arange(labels.shape[0], device=labels.device)
    original_true = probabilities[indices, labels]
    related_true = related_probabilities[indices, labels]
    control_true = control_probabilities[indices, labels]

    related_drop = original_true - related_true
    control_drop = torch.abs(original_true - control_true)
    valid = labels != 0
    if valid.any():
        ranking = F.relu(margin - related_drop[valid] + control_drop[valid]).mean()
        relevance = anchor[valid]
        energy = original["channel_energy"][valid]
        alignment = 1.0 - F.cosine_similarity(energy, relevance, dim=1).mean()
    else:
        ranking = original_true.sum() * 0.0
        alignment = ranking

    return {
        "logits": original["logits"],
        "ranking_loss": ranking,
        "alignment_loss": alignment,
        "related_drop": related_drop.detach(),
        "control_drop": control_drop.detach(),
    }


@torch.no_grad()
def counterfactual_evidence(
    model,
    inputs,
    predicted,
    rpms,
    bandwidth=0.006,
    original_logits=None,
    original_tf_response=None,
):
    original = (
        original_logits if original_logits is not None else model(inputs)
    )
    anchor = physical_anchor_weights(
        predicted,
        rpms,
        model.center_frequencies,
        model.fault_ratios,
        model.sample_rate,
        bandwidth=bandwidth,
    )
    control = control_weights(anchor)
    if original_tf_response is None:
        related = model(inputs, channel_mask=1.0 - anchor)
        control_logits = model(inputs, channel_mask=1.0 - control)
    else:
        related = model.classify_tf(
            original_tf_response * (1.0 - anchor).unsqueeze(-1)
        )[0]
        control_logits = model.classify_tf(
            original_tf_response * (1.0 - control).unsqueeze(-1)
        )[0]
    probability = F.softmax(original, dim=1)
    related_probability = F.softmax(related, dim=1)
    control_probability = F.softmax(control_logits, dim=1)
    indices = torch.arange(predicted.shape[0], device=predicted.device)
    confidence = probability[indices, predicted]
    related_drop = confidence - related_probability[indices, predicted]
    control_change = torch.abs(confidence - control_probability[indices, predicted])
    evidence = confidence + related_drop - control_change
    return evidence, confidence, related_drop, control_change
