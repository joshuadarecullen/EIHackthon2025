from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import clip
from transformers import CLIPModel

# --- Resize Positional Embeddings for 120x160 input ---
def resize_pos_embed(model, new_grid_size=(7, 10)):  # (H, W)
    old_pe = model.positional_embedding  # (1, num_patches+1, dim)
    cls_token = old_pe[:, 0:1, :]
    patch_pe = old_pe[:, 1:, :]

    old_grid_size = int(patch_pe.shape[1] ** 0.5)  # typically 14
    patch_pe = patch_pe.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=new_grid_size, mode='bilinear')
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, patch_pe.shape[1])

    new_pe = torch.cat([cls_token, patch_pe], dim=1)
    model.positional_embedding = torch.nn.Parameter(new_pe)

class MetLitModule(LightningModule):
    """LightningModule for finetuing a CLIP model to Meteorological data.
        - Geopotential
        - Air pressure
        - Temperature

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_type: str= "openai/clip-vit-base-patch32",
    ) -> None:
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Load pretrained clip model
        self.model = CLIPModel.from_pretrained("model_type")
        resize_pos_embed(self.model.visual, new_grid_size=(7, 10))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        met_data, texts = batch

        # Forward pass
        logits_per_met, logits_per_text = self.model(met_data, texts)

        """
        Compute loss:
        Ground truth  acts as the target labels when comparing each image with the corresponding text. It assumes that:

            Metdata i is paired with Text i

        There is a one-to-one correspondence between the metdata and texts in each batch

        CLIP models compute a similarity matrix between all met_data and all texts in the batch.
        """
        ground_truth = torch.arange(len(met_data), dtype=torch.float32)
        loss = (self.loss_met_data(logits_per_met, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of met data and text
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of met data and texts.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass


    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Convert model to fp32 if the helper exists
        if self.device.type != "cpu" and hasattr(self, 'convert_models_to_fp32'):
            self.convert_models_to_fp32(self.model)

    def on_after_backward(self):
        # Convert model weights back to fp16 if needed
        if self.device.type != "cpu" and hasattr(self.model, "convert_weights"):
            self.model.convert_weights()

    def convert_models_to_fp32(self, model):
        """Recursively converts model to float32 for stability (CLIP-specific helper)"""
        for p in model.parameters():
            if p.dtype == torch.float16:
                p.data = p.data.float()
            if p.grad is not None and p.grad.dtype == torch.float16:
                p.grad.data = p.grad.data.float()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
