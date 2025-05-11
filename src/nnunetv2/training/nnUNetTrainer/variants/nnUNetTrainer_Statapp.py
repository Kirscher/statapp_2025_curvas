"""
Custom trainer for the statapp application.

This module provides a custom trainer for nnunet with stop early and custom network initialization.

I also added emoji and color formatting because why not :)
"""
import os
import sys
from datetime import datetime
from time import time, sleep
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from rich.text import Text

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from statapp.utils.utils import setup_logging, info, pretty_print


class nnUNetTrainer_Statapp(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        self.statapp_logger = setup_logging(False)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.early_stopping_patience = 20
        self.epochs_without_improvement = 0

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """
        Print to log file and optionally to console using fancy formatting.

        Args:
            *args: The message components to log (strings or Text objects)
            also_print_to_console: Whether to also print to console
            add_timestamp: Whether to add a timestamp (not used, kept for compatibility)
        """
        if self.local_rank == 0:
            # Check if any arg is a Text object
            has_text_object = any(isinstance(arg, Text) for arg in args)

            if has_text_object:
                # If we have a Text object, we need to convert it to string for logging
                # but keep the original for console output
                text_for_console = args[0] if len(args) == 1 else Text.assemble(*args)
                message = str(text_for_console)
            else:
                # Format the message as before
                message = " ".join(str(arg) for arg in args)
                text_for_console = message

            # Log to file
            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    self.statapp_logger.debug(message)
                    successful = True
                except IOError:
                    self.statapp_logger.error(f"Failed to log: {sys.exc_info()}")
                    sleep(0.5)
                    ctr += 1

            # Print to console with fancy formatting if requested
            if also_print_to_console:
                pretty_print(text_for_console)
        elif also_print_to_console:
            # Check if any arg is a Text object
            has_text_object = any(isinstance(arg, Text) for arg in args)

            if has_text_object:
                # If we have a Text object, use it directly
                text_for_console = args[0] if len(args) == 1 else Text.assemble(*args)
            else:
                # Format the message as before
                text_for_console = " ".join(str(arg) for arg in args)

            pretty_print(text_for_console)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        See the super method documentation. This work by first getting the original network initialized by nnUNet and
        the dynamic-network-architectures library, then apply a seeded init weights, using the SEED
        variable environment as the manual seed of the torch generator.

        Please note that this only works for the unet architecture of the library, as others may have different
        initialization methods. Look at https://github.com/MIC-DKFZ/dynamic-network-architectures for more information.
        """
        # We first get the network initialized by https://github.com/MIC-DKFZ/dynamic-network-architectures lib.
        network_architecture = nnUNetTrainer.build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels, enable_deep_supervision)
        # Then we apply the seeded init.
        if 'SEED' in os.environ:
            network_architecture.apply(SeededInitWeights_He(seed=os.environ.get('SEED')))
        return network_architecture

    def on_epoch_end(self):
        """
        Called at the end of each epoch to log metrics and handle checkpointing.
        """
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Format metrics with fancy formatting
        train_loss = np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4)
        val_loss = np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4)
        dice_values = [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]]
        epoch_time = np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - 
                             self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)

        # Log metrics with fancy formatting and colors
        self.print_to_log_file(Text.assemble(
            ("📊 ", ""),
            ("Epoch ", "bold"),
            (f"{self.current_epoch} ", "bold cyan"),
            ("Metrics:", "bold magenta")
        ))
        self.print_to_log_file(Text.assemble(
            ("   🔴 ", ""),
            ("Training Loss: ", "bold red"),
            (f"{train_loss}", "bold white")
        ))
        self.print_to_log_file(Text.assemble(
            ("   🔵 ", ""),
            ("Validation Loss: ", "bold blue"),
            (f"{val_loss}", "bold white")
        ))
        self.print_to_log_file(Text.assemble(
            ("   🎯 ", ""),
            ("Pseudo Dice Scores: ", "bold green"),
            (f"{dice_values}", "bold white")
        ))
        self.print_to_log_file(Text.assemble(
            ("   ⏱️ ", ""),
            ("Epoch Time: ", "bold yellow"),
            (f"{epoch_time}", "bold white"),
            (" seconds", "")
        ))

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_latest.pth'))
            self.print_to_log_file(Text.assemble(
                ("💾 ", ""),
                ("Saved periodic checkpoint at epoch ", "bold green"),
                (f"{current_epoch}", "bold cyan")
            ))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            best_ema_rounded = np.round(self._best_ema, decimals=4)
            self.print_to_log_file(Text.assemble(
                ("🎉 ", ""),
                ("New best EMA pseudo Dice: ", "bold green"),
                (f"{best_ema_rounded}", "bold cyan"),
                ("! Saving checkpoint...", "bold green")
            ))
            self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_best.pth'))
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            self.print_to_log_file(Text.assemble(
                ("⏳ ", ""),
                ("No improvement for ", "bold yellow"),
                (f"{self.epochs_without_improvement}", "bold red"),
                (" epochs. Early stopping at ", "bold yellow"),
                (f"{self.early_stopping_patience}", "bold red"),
                (" epochs without improvement.", "bold yellow")
            ))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


    def run_training(self):
        """
        Run the training loop with early stopping.
        """
        self.print_to_log_file(Text.assemble(
            ("🚀 ", ""),
            ("Starting training with ", "bold blue"),
            (f"{self.num_epochs}", "bold cyan"),
            (" maximum epochs", "bold blue")
        ))
        self.print_to_log_file(Text.assemble(
            ("🛑 ", ""),
            ("Early stopping patience set to ", "bold yellow"),
            (f"{self.early_stopping_patience}", "bold red"),
            (" epochs", "bold yellow")
        ))

        if 'SEED' in os.environ:
            self.print_to_log_file(Text.assemble(
                ("🌱 ", ""),
                ("Using random seed: ", "bold green"),
                (f"{os.environ.get('SEED')}", "bold cyan")
            ))

        self.on_train_start()
        epoch = self.current_epoch

        while epoch < self.num_epochs and self.epochs_without_improvement < self.early_stopping_patience:
            self.print_to_log_file(Text.assemble(
                ("⏳ ", ""),
                ("Starting epoch ", "bold blue"),
                (f"{epoch}", "bold cyan"),
                ("/", ""),
                (f"{self.num_epochs}", "bold cyan")
            ))
            self.on_epoch_start()

            # Training phase
            self.print_to_log_file(Text.assemble(
                ("🏋️ ", ""),
                ("Training phase...", "bold green")
            ))
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            # Validation phase
            self.print_to_log_file(Text.assemble(
                ("🔍 ", ""),
                ("Validation phase...", "bold blue")
            ))
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Check for early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.print_to_log_file(Text.assemble(
                    ("🛑 ", ""),
                    ("Early stopping triggered at epoch ", "bold red"),
                    (f"{epoch}", "bold cyan"),
                    ("!", "bold red")
                ))
                self.print_to_log_file(Text.assemble(
                    ("   ", ""),
                    ("No improvement for ", "bold yellow"),
                    (f"{self.early_stopping_patience}", "bold red"),
                    (" consecutive epochs.", "bold yellow")
                ))

        self.print_to_log_file(Text.assemble(
            ("✅ ", ""),
            ("Training completed after ", "bold green"),
            (f"{epoch}", "bold cyan"),
            (" epochs", "bold green")
        ))
        self.on_train_end()




class SeededInitWeights_He(object):
    """
    Initialize network weights using He initialization with a fixed random seed.

    This allows for reproducible weight initialization when the SEED environment variable is set.
    """
    def __init__(self, neg_slope: float = 1e-2, seed: int=None):
        """
        Initialize the weight initializer.

        Args:
            neg_slope: Negative slope for the LeakyReLU
            seed: Random seed for reproducibility
        """
        self.neg_slope = neg_slope
        self.generator = torch.Generator()
        if seed:
            self.generator = self.generator.manual_seed(int(seed))

    def __call__(self, module):
        """
        Apply He initialization to the module weights.

        Args:
            module: The module to initialize
        """
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope, generator=self.generator)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
