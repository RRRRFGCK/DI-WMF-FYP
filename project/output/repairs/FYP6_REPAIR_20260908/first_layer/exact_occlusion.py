"""Vectorised replacement of only the original per-image occlusion loop.

Same RNG draws, integer boxes, per-image/channel mean, and clipping. Tested for
bitwise equality of both output pixels and the post-call generator state.
"""
import math
import torch


def vectorized_occlusion(pixels, severity, generator):
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")
    if severity == 0:
        return pixels.clone()
    squeeze = pixels.ndim == 3
    if squeeze:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4:
        raise ValueError("pixels must have shape [C,H,W] or [B,C,H,W]")
    batch, _, height, width = pixels.shape
    side_fraction = math.sqrt(severity)
    box_h = max(1, min(height, int(round(height * side_fraction))))
    box_w = max(1, min(width, int(round(width * side_fraction))))
    tops = torch.randint(0, height-box_h+1, (batch,), generator=generator, device=pixels.device)
    lefts = torch.randint(0, width-box_w+1, (batch,), generator=generator, device=pixels.device)
    yy = torch.arange(height, device=pixels.device).view(1, height, 1)
    xx = torch.arange(width, device=pixels.device).view(1, 1, width)
    mask = ((yy >= tops[:, None, None]) & (yy < tops[:, None, None] + box_h)
            & (xx >= lefts[:, None, None]) & (xx < lefts[:, None, None] + box_w))
    fill = pixels.mean(dim=(2, 3), keepdim=True)
    corrupted = torch.where(mask[:, None], fill, pixels).clamp(0.0, 1.0)
    return corrupted.squeeze(0) if squeeze else corrupted


def self_test(original, device):
    results = []
    for channels, size in [(1, 20), (3, 32), (3, 64)]:
        for batch in [1, 17, 1024]:
            pixel_rng = torch.Generator(device=device).manual_seed(12345)
            pixels = torch.rand((batch, channels, size, size), device=device, generator=pixel_rng)
            for severity in [0.0, .05, .1, .2, .3, 1.0]:
                for seed in ([0, 2] if batch < 1024 else [1]):
                    a = torch.Generator(device=device).manual_seed(seed)
                    b = torch.Generator(device=device).manual_seed(seed)
                    reference = original(pixels, "occlusion", severity, a)
                    repaired = vectorized_occlusion(pixels, severity, b)
                    same = torch.equal(reference, repaired)
                    same_rng = torch.equal(a.get_state(), b.get_state())
                    assert same and same_rng
                    results.append({"batch": batch, "channels": channels, "size": size,
                                    "severity": severity, "noise_seed": seed,
                                    "pixels_bitwise_equal": same, "rng_state_bitwise_equal": same_rng})
    return results
