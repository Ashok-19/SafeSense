import torch
from safesense import SafeSenseFactorized, parameter_count


def test_parameter_count_and_shapes():
    model = SafeSenseFactorized().eval()
    assert parameter_count(model) == 440_562
    skeleton = torch.zeros(2, 128, 177)
    imu = torch.zeros(2, 128, 12)
    mask = torch.ones(2, 128, 2)
    with torch.inference_mode():
        logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (2, 27)
    assert weights.shape == (2, 2)


def test_availability_weights_follow_masks():
    model = SafeSenseFactorized().eval()
    skeleton = torch.zeros(2, 128, 177)
    imu = torch.zeros(2, 128, 12)
    mask = torch.ones(2, 128, 2)
    mask[0, :, 1] = 0
    mask[1, :, 0] = 0
    with torch.inference_mode():
        _, weights = model(skeleton, imu, mask)
    assert torch.equal(weights[0], torch.tensor([1.0, 0.0]))
    assert torch.equal(weights[1], torch.tensor([0.0, 1.0]))
