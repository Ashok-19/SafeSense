from safesense.training import TrainingConfig


def test_paper_defaults():
    c = TrainingConfig()
    assert c.seed == 22
    assert c.epochs == 60
    assert c.batch_size == 32
    assert c.learning_rate == 1e-3
    assert c.weight_decay == 1e-4
    assert c.swa_start_epoch == 45
