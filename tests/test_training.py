"""
Tests for training pipeline.
"""

from pathlib import Path

import torch


class TestModels:
    """Tests for model creation and forward pass."""

    def test_vit_classifier_creation(self):
        """Test creating ViT classifier."""
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        assert model.num_classes == 10
        assert model.image_size == 32

    def test_vit_forward(self):
        """Test ViT forward pass."""
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=5, num_channels=3, image_size=32)

        x = torch.randn(4, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (4, 5)

    def test_mim_model_creation(self):
        """Test creating MIM model."""
        from vision_spectra.models import MIMModel, create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        encoder = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        mim_model = MIMModel(
            encoder=encoder,
            decoder_embed_dim=64,
            decoder_depth=2,
            decoder_num_heads=2,
            mask_ratio=0.75,
        )

        assert mim_model.mask_ratio == 0.75

    def test_mim_forward(self):
        """Test MIM forward pass."""
        from vision_spectra.models import MIMModel, create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        encoder = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        mim_model = MIMModel(
            encoder=encoder,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=2,
            mask_ratio=0.75,
        )

        x = torch.randn(4, 3, 32, 32)
        loss, pred, mask = mim_model(x)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert mask.shape[0] == 4  # Batch size

    def test_multitask_model(self):
        """Test multitask model."""
        from vision_spectra.models import MultitaskViT, create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        encoder = create_vit_classifier(config, num_classes=5, num_channels=3, image_size=32)

        mtl_model = MultitaskViT(
            encoder=encoder,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=2,
            mask_ratio=0.5,
        )

        x = torch.randn(4, 3, 32, 32)

        # Test classification mode
        logits = mtl_model(x, mode="classification")
        assert logits.shape == (4, 5)

        # Test MIM mode
        loss, pred, mask = mtl_model(x, mode="mim")
        assert torch.isfinite(loss)

        # Test multitask mode
        logits, mim_loss, pred, mask = mtl_model(x, mode="multitask")
        assert logits.shape == (4, 5)
        assert torch.isfinite(mim_loss)


class TestSmokeTraining:
    """Smoke tests for training pipeline."""

    def test_smoke_classification(self):
        """Smoke test for classification training."""
        from vision_spectra.data import get_dataset
        from vision_spectra.losses import get_loss
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import DatasetName, ExperimentConfig, set_seed
        from vision_spectra.training import ClassificationTrainer

        set_seed(42)

        # Create minimal config
        config = ExperimentConfig(
            name="test_cls",
            seed=42,
            device="cpu",
        )
        config.dataset.name = DatasetName.SYNTHETIC
        config.dataset.batch_size = 4
        config.dataset.num_samples_train = 16
        config.dataset.num_samples_val = 8
        config.dataset.num_samples_test = 8
        config.dataset.num_workers = 0
        config.training.epochs = 2
        config.training.smoke_test = True
        config.training.early_stopping = False
        config.spectral.enabled = False

        # Create dataset
        data_dir = Path("/tmp/vision_spectra_test")
        dataset = get_dataset(config.dataset, data_dir)
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        info = dataset.get_info()

        # Create model
        model = create_vit_classifier(
            config.model,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
            image_size=info.image_size[0],
        )

        # Create loss
        criterion = get_loss(config.loss)

        # Create trainer
        trainer = ClassificationTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
        )

        # Train one epoch manually
        train_metrics = trainer.train_epoch()

        assert "loss" in train_metrics
        assert "accuracy" in train_metrics
        assert train_metrics["loss"] > 0

    def test_smoke_mim(self):
        """Smoke test for MIM pretraining."""
        from vision_spectra.data import get_dataset
        from vision_spectra.models import MIMModel, create_vit_classifier
        from vision_spectra.settings import DatasetName, ExperimentConfig, set_seed
        from vision_spectra.training import MIMTrainer

        set_seed(42)

        config = ExperimentConfig(
            name="test_mim",
            seed=42,
            device="cpu",
        )
        config.dataset.name = DatasetName.SYNTHETIC
        config.dataset.image_size = 32  # Must be divisible by patch_size (16)
        config.dataset.batch_size = 4
        config.dataset.num_samples_train = 16
        config.dataset.num_samples_val = 8
        config.dataset.num_samples_test = 8
        config.dataset.num_workers = 0
        config.model.mask_ratio = 0.5
        config.training.epochs = 2
        config.training.smoke_test = True
        config.training.early_stopping = False
        config.spectral.enabled = False

        data_dir = Path("/tmp/vision_spectra_test")
        dataset = get_dataset(config.dataset, data_dir)
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        info = dataset.get_info()

        # Create encoder and MIM model
        encoder = create_vit_classifier(
            config.model,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
            image_size=info.image_size[0],
        )

        model = MIMModel(
            encoder=encoder,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=2,
            mask_ratio=config.model.mask_ratio,
        )

        trainer = MIMTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_channels=info.num_channels,
        )

        train_metrics = trainer.train_epoch()

        assert "loss" in train_metrics
        assert train_metrics["loss"] > 0

    def test_smoke_multitask(self):
        """Smoke test for multitask training."""
        from vision_spectra.data import get_dataset
        from vision_spectra.losses import get_loss
        from vision_spectra.models import MultitaskViT, create_vit_classifier
        from vision_spectra.settings import DatasetName, ExperimentConfig, set_seed
        from vision_spectra.training import MultitaskTrainer

        set_seed(42)

        config = ExperimentConfig(
            name="test_mtl",
            seed=42,
            device="cpu",
        )
        config.dataset.name = DatasetName.SYNTHETIC
        config.dataset.image_size = 32  # Must be divisible by patch_size (16)
        config.dataset.batch_size = 4
        config.dataset.num_samples_train = 16
        config.dataset.num_samples_val = 8
        config.dataset.num_samples_test = 8
        config.dataset.num_workers = 0
        config.model.mask_ratio = 0.5
        config.loss.mtl_cls_weight = 1.0
        config.loss.mtl_mim_weight = 0.5
        config.training.epochs = 2
        config.training.smoke_test = True
        config.training.early_stopping = False
        config.spectral.enabled = False

        data_dir = Path("/tmp/vision_spectra_test")
        dataset = get_dataset(config.dataset, data_dir)
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        info = dataset.get_info()

        encoder = create_vit_classifier(
            config.model,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
            image_size=info.image_size[0],
        )

        model = MultitaskViT(
            encoder=encoder,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=2,
            mask_ratio=config.model.mask_ratio,
        )

        cls_criterion = get_loss(config.loss)

        trainer = MultitaskTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cls_criterion=cls_criterion,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
        )

        train_metrics = trainer.train_epoch()

        assert "loss" in train_metrics
        assert "cls_loss" in train_metrics
        assert "mim_loss" in train_metrics


class TestCheckpointing:
    """Tests for checkpointing."""

    def test_save_and_load(self):
        """Test saving and loading checkpoint."""
        import tempfile

        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig
        from vision_spectra.utils import load_checkpoint, save_checkpoint

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=5, num_channels=3, image_size=32)

        # Get initial output
        x = torch.randn(2, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            out1 = model(x).clone()

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_checkpoint(Path(f.name), model, epoch=5)
            ckpt_path = Path(f.name)

        # Create new model and load
        model2 = create_vit_classifier(config, num_classes=5, num_channels=3, image_size=32)

        result = load_checkpoint(ckpt_path, model2)

        assert result["epoch"] == 5

        # Outputs should match
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1, out2)

        # Cleanup
        ckpt_path.unlink()


class TestReproducibility:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self):
        """Test that setting seed gives reproducible results."""
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig, set_seed

        def create_model():
            set_seed(42)
            config = ModelConfig(name="vit_tiny_patch16_224")
            return create_vit_classifier(config, num_classes=5, num_channels=3, image_size=32)

        model1 = create_model()
        model2 = create_model()

        # Weights should be identical
        for (n1, p1), (_n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters(), strict=False
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"
