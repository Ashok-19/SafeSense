import numpy as np
from safesense.data import feature_arrays


def test_feature_dimensions():
    fused = np.zeros((3, 1, 128, 22, 3), dtype=np.float32)
    fused[:, 0, :, :20] = 1.0
    fused[:, 0, :10, 20:] = 2.0
    skeleton, imu, mask = feature_arrays(fused)
    assert skeleton.shape == (3, 128, 177)
    assert imu.shape == (3, 128, 12)
    assert mask.shape == (3, 128)
    assert np.all(mask[:, :10] == 1)
    assert np.all(mask[:, 10:] == 0)
