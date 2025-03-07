import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_convpixelformer_from_plans 
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.utilities.helpers import empty_cache, dummy_context

class nnUNetTrainer_ConvPixelPixelFormer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
  
        self.is_ddp = dist.is_available() and dist.is_initialized()
        print("dist.is_available() ", dist.is_available(), "dist.is_initialized()",dist.is_initialized() )
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        # print("self.configuration_manager", self.configuration_manager)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None
                
        # split learning rate for conv layer and transformer layer
        self.split_lr = False
        # freeze encoder? TODO
        self.freeze_encoder = False


        # assert 'Lin' in self.configuration_manager.network_arch_class_name, f'Only Lin attetn network architecture is supported. Got {self.configuration_manager.network_arch_class_name}'
        # 1) possible 3d_fullres_plain_unet_small 
        # 2) possible 3d_fullres_plain_unet_resenc_small
        # 3) possible 3d_fullres_plain_convformer_small

        if 'plain_unet' in configuration:
            # we use the original unet copy from https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/models/pytorch/segmentation/unet.py 
            assert 'BasicUNet' in self.configuration_manager.network_arch_class_name, f'Only UNet3D network architecture is supported. Got {self.configuration_manager.network_arch_class_name}'
            self.model_type = 'nnUNet_UNet'
            self.initial_lr = 1e-3
            self.weight_decay = 3e-5
            self.oversample_foreground_percent = 0.33
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            self.num_epochs = 1000
            self.current_epoch = 0
            self.enable_deep_supervision = False
            self.optimizer_type = 'SGD'
            self.lr_scheduler_type = 'Poly'
        
            print("=====================================================================")
            print("Using Plain Unet default parameters")

            if '_epoch' in configuration or '_iter' in configuration:
                parts = configuration.split('_')
                for p in parts:
                    if 'epoch' in p:
                        self.num_epochs = int(p.replace('epoch', ''))
                    if 'iter' in p:
                        self.num_iterations_per_epoch = int(p.replace('iter', ''))
            else:
                self.num_epochs = 1000
                self.num_iterations_per_epoch = 250       

            self.current_epoch = 0
        elif 'convformer' in configuration:
            print("configuration", configuration)

            assert 'ConvPixelInceptionFormer' in self.configuration_manager.network_arch_class_name, f'Only ConvPixelInceptionFormer network architecture is supported. Got {self.configuration_manager.network_arch_class_name}'
            self.model_type = 'ConvPixelInceptionFormer'
            # if "pos_2_conv" in configuration:
            #     self.initial_lr = 8e-4
            #     print("Using convolution learning rate")
            # else:
            self.initial_lr = 5e-5

            self.weight_decay = 3e-5
            self.oversample_foreground_percent = 0.33
            self.num_val_iterations_per_epoch = 50
            self.optimizer_type = 'AdamW'
            self.lr_scheduler_type = 'Cosine'
            self.warmup_epochs = 10
            self.current_epoch = 0
            print("=====================================================================")
            print("Using Convpixelformer parameters")

            if len(self.configuration_manager.network_arch_init_kwargs['deep_supervision_scales']) == 0:
                self.enable_deep_supervision = False
            else:
                self.enable_deep_supervision = True
                print("====== enable_deep_supervision in Convpixelformer =========", self.enable_deep_supervision)


            if '_epoch' in configuration or '_iter' in configuration:
                parts = configuration.split('_')
                print("====== Using adjusted num_epochs and num_iterations_per_epoch in Video MAE =========")
                for p in parts:
                    if 'epoch' in p:
                        self.num_epochs = int(p.replace('epoch', ''))
                        print("====== num_epochs in Convpixelformer =========", self.num_epochs)
                    elif 'iter' in p:
                        self.num_iterations_per_epoch = int(p.replace('iter', ''))
                        print("====== num_iterations_per_epoch in Convpixelformer =========", self.num_iterations_per_epoch)
  
            else:
                self.num_epochs = 1000
                self.num_iterations_per_epoch = 250

        else:
            raise NotImplementedError(f"Configuration {configuration} not implemented")

            
        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
  

        return get_network_convpixelformer_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)


    def configure_optimizers(self):
        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
        elif self.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, betas=(0.9, 0.95))
  
        else:
            raise ValueError(f"Unknown optimizer type {self.optimizer_type}")

        if self.lr_scheduler_type == 'Poly':
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        elif self.lr_scheduler_type == 'Cosine':
            import transformers
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=self.warmup_epochs, num_training_steps=self.num_epochs)
        else:
            raise ValueError(f"Unknown lr scheduler type {self.lr_scheduler_type}")

        # # hook
        # for name, param in self.network.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(lambda grad, n=name: print(n, grad.mean(), grad.min(), grad.max()))


        return optimizer, lr_scheduler


    def train_step(self, batch: dict) -> dict:
        # self.print_to_log_file(f'Train step start')
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        # print loss and check if NaN or inf
        if torch.isnan(l).any():
            print("=====================================")
            print("Loss is NaN")
        
        if torch.isinf(l).any():
            print("=====================================")
            print("Loss is inf")
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 11.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 11.0)
            self.optimizer.step()
        # self.print_to_log_file(f'Train loss: {l.detach().cpu().numpy()}')
        return {'loss': l.detach().cpu().numpy()}
