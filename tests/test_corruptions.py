import numpy as np
from safesense.corruptions import CorruptionSpec, apply_corruption, scenario_specs


def dummy_fused():
    x = np.zeros((2, 1, 128, 22, 3), dtype=np.float32)
    x[:, 0, :8, :20] = 1.0
    x[:, 0, :8, 20:] = 2.0
    return x


def test_suite_has_one_clean_plus_24_corruptions():
    specs = scenario_specs()
    assert len(specs) == 25
    assert specs[0].name == 'clean'
    assert {s.category for s in specs[1:]} == {'shift', 'frame_loss', 'noise', 'missing', 'composite'}


def test_shift_is_zero_padded_not_circular():
    x = dummy_fused()
    spec = CorruptionSpec('imu_shift_+4', 'shift', 'imu', shift=4)
    y, masks = apply_corruption(x, spec, seed=1, scales={'skeleton': 1.0, 'imu': 1.0})
    assert np.all(y[:, 0, :4, 20:] == 0)
    assert not masks['imu'][:, :4].any()


def test_corruption_is_deterministic_given_seed():
    x = dummy_fused()
    spec = CorruptionSpec('imu_frame_loss_30', 'frame_loss', 'imu', loss_fraction=0.3)
    a, ma = apply_corruption(x, spec, seed=123, scales={'skeleton': 1.0, 'imu': 1.0})
    b, mb = apply_corruption(x, spec, seed=123, scales={'skeleton': 1.0, 'imu': 1.0})
    assert np.array_equal(a, b)
    assert np.array_equal(ma['imu'], mb['imu'])
