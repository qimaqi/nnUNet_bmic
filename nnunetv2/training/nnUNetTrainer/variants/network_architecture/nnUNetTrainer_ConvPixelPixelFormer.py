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
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels

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
            self.mixed_precision = True
        
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

            # assert 'ConvPixelInceptionFormer' in self.configuration_manager.network_arch_class_name, f'Only ConvPixelInceptionFormer network architecture is supported. Got {self.configuration_manager.network_arch_class_name}'
            self.model_type = 'ConvPixelInceptionFormer'

            self.initial_lr = 5e-5
            if '_learningrate' in configuration:
                parts = configuration.split('_')
                for p in parts:
                    if 'learningrate' in p:
                        self.initial_lr = float(p.replace('learningrate', ''))
                        print("====== initial_lr in ConvPixelInceptionFormer =========", self.initial_lr)


            # self.weight_decay = 3e-5
            self.oversample_foreground_percent = 0.33
            self.num_val_iterations_per_epoch = 50
            self.optimizer_type = 'AdamW'
            self.weight_decay = 3e-5
            self.lr_scheduler_type = 'Cosine'
            self.warmup_epochs = 10
            self.current_epoch = 0
            self.mixed_precision = False
            if '_mixed' in configuration:
                self.mixed_precision = True

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
        if self.device.type == 'cuda':
            if self.mixed_precision:
                self.grad_scaler = GradScaler() 
            else:
                self.grad_scaler = None
        else:
            self.grad_scaler = None
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
            optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          betas=(0.9, 0.999), eps=1e-4)
  
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
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
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

        # print loss and check if NaN or inf
        # if torch.isnan(l).any():
        #     print("=====================================")
        #     print("Loss is NaN")
        
        # if torch.isinf(l).any():
        #     print("=====================================")
        #     print("Loss is inf")
        
        if self.grad_scaler is not None:
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)
                # del data
                l = self.loss(output, target)
                
            backward_start_t = time()
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # print(f"backward time with grad_scaler: {time() - backward_start_t}")
        else:
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            backward_start_t = time()
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            # print(f"backward time without grad_scaler: {time() - backward_start_t}")
        # self.print_to_log_file(f'Train loss: {l.detach().cpu().numpy()}')
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'loss': l.detach().cpu().numpy()}
        

    def perform_actual_validation(self, save_probabilities: bool = False):
        print("Performing actual validation")
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False, mixed_precision=self.mixed_precision)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
