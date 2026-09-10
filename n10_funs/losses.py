"""Joint classification + detection + re-ID loss (Eq. 1/2/5 of the paper)."""
import torch
import torch.nn.functional as F


def focal_classification_loss(pred, target, alpha=2.0, beta=4.0):
    """Eq. 1. pred: (B, N_CLASSES, H, W) already sigmoid-ed. target: (B, H, W) in [0,1] (1 at
    exact centers, decaying via the Gaussian elsewhere, 0 far from any object). With
    N_CLASSES=1 this is a plain cow-vs-background focal loss."""
    pred = pred.squeeze(1).clamp(1e-6, 1 - 1e-6)
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()
    pos_loss = -pos_mask * (1 - pred) ** alpha * torch.log(pred)
    neg_loss = -neg_mask * (1 - target) ** beta * pred ** alpha * torch.log(1 - pred)
    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def detection_l1_loss(pred_offset, pred_size, offset_gt, size_gt, centers, mask, lambda_s=0.1):
    """Eq. 2, gathered at GT centers only (paper: 'the following l1 losses for the two heads')."""
    B, _, H, W = pred_offset.shape
    flat_off = pred_offset.permute(0, 2, 3, 1).reshape(B, H * W, 2)
    flat_size = pred_size.permute(0, 2, 3, 1).reshape(B, H * W, 2)
    idx = (centers[..., 1] * W + centers[..., 0]).clamp(0, H * W - 1)
    got_off = torch.gather(flat_off, 1, idx.unsqueeze(-1).expand(-1, -1, 2))[mask]
    got_size = torch.gather(flat_size, 1, idx.unsqueeze(-1).expand(-1, -1, 2))[mask]
    if got_off.numel() == 0:
        z = torch.zeros((), device=pred_offset.device)
        return z, z
    off_loss = F.l1_loss(got_off, offset_gt[mask])
    size_loss = F.l1_loss(got_size, size_gt[mask])
    return off_loss, lambda_s * size_loss


def joint_loss(model, outputs, heatmap_gt, offset_gt, size_gt, centers, identity_gt, mask):
    """Returns (total_loss, dict_of_components) for logging."""
    l_classification = focal_classification_loss(outputs["classification"], heatmap_gt)
    l_off, l_size = detection_l1_loss(outputs["offset"], outputs["size"], offset_gt, size_gt, centers, mask)
    l_detection = l_off + l_size

    reid_embed = model.gather_embeddings(outputs["reid_map"], centers, mask)
    if reid_embed.numel() > 0:
        identity_logits = model.identity_fc(reid_embed)
        l_identity = F.cross_entropy(identity_logits, identity_gt[mask])
    else:
        l_identity = torch.zeros((), device=heatmap_gt.device)

    w1, w2, w3 = model.log_vars[0], model.log_vars[1], model.log_vars[2]
    total = (1.0 / 3.0) * (
        torch.exp(-w1) * l_classification + torch.exp(-w2) * l_detection + torch.exp(-w3) * l_identity
        + w1 + w2 + w3
    )
    parts = {
        "classification": l_classification.item(), "off": l_off.item(), "size": l_size.item(),
        "identity": l_identity.item(), "total": total.item(),
        "w1": w1.item(), "w2": w2.item(), "w3": w3.item(),
    }
    return total, parts
