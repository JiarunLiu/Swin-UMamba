import os
from os.path import join

import torch
from torch import device, nn
from torch._C import device
from torchinfo import summary
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.SwinUMambaD import get_swin_umamba_d_from_plans


class nnUNetTrainerSwinUMambaD(nnUNetTrainer):

    """ Swin-UMamba$\dagger$ with Mamba-based decoder"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 10

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True) -> nn.Module:

        model = get_swin_umamba_d_from_plans(
            plans_manager, 
            dataset_json, 
            configuration_manager,
            num_input_channels, 
            deep_supervision=enable_deep_supervision, 
            use_pretrain=True
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr, 
            weight_decay=self.weight_decay, 
            eps=1e-5,
            betas=(0.9, 0.999),
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def on_epoch_end(self):
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch}.pth'))
        super().on_epoch_end()

    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
        if self.current_epoch < self.freeze_encoder_epochs:
            self.print_to_log_file("Freezing the encoder")
            if self.is_ddp:
                self.network.module.freeze_encoder()
            else:
                self.network.freeze_encoder()
        else:
            self.print_to_log_file("Unfreezing the encoder")
            if self.is_ddp:
                self.network.module.unfreeze_encoder()
            else:
                self.network.unfreeze_encoder()
        super().on_train_epoch_start()


    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0,1.0], [0.25,0.25], [0.125,0.125], [0.0625, 0.0625]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales