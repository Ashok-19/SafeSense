from pathlib import Path
import torch
from safesense import SafeSenseFactorized, parameter_count


def test_released_checkpoint_loads_strictly():
    checkpoint = Path('checkpoints/safesense_seed22_swa.pt')
    assert checkpoint.is_file()
    model = SafeSenseFactorized()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    assert parameter_count(model) == 440_562
