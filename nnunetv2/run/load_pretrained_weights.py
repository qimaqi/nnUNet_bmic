import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import omegaconf

def interpolate_pretrained_pos_enc_decoder(args: dict, state_dict: dict) -> dict:
    """
    Adjusts the pretrained positional encoding tensor to fit the current model's dimensions.(larger)

    Args:
        args (dict): The input arguments to the model
        state_dict (dict): The loaded state dictionary to adjust

    Returns:
        dict: The adjusted state dictionary with the updated positional encoding
    """
    orig_patches_per_dim = 224 // 16  # original 224x224 model with patch size 16
    new_patches_per_dim = args.img_size // 16
    temporal_patches = 8
    new_patches_video_dim = args.num_frames // args.t_patch_size
    if orig_patches_per_dim != new_patches_per_dim:
        
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = new_patches_per_dim + 0.1, new_patches_per_dim + 0.1
        # reshape to have temporal separation
        pos_enc = state_dict["decoder_pos_embed"]
        # TODO we interpolate only the spatial but not time
        
        pos_enc = pos_enc.reshape(
            1, temporal_patches, orig_patches_per_dim**2, -1
        )
        print("pos decoder embed before interpolate", pos_enc.size())
        # reshape to also have spatial separation
        pos_enc = pos_enc.reshape(
            1, temporal_patches, orig_patches_per_dim, orig_patches_per_dim, -1
        )
        dim = pos_enc.shape[-1]
        # we only do interpolate on frame not temporal
        # print("decoder pos_enc to interpolate ", pos_enc.size())
        pos_enc = pos_enc.permute(0, 4, 1, 2, 3)
        # print("decoder pos_enc", pos_enc.size())
        pos_enc = torch.nn.functional.interpolate(
            pos_enc,
            size=(new_patches_video_dim ,new_patches_per_dim, new_patches_per_dim),
            mode="area"
        )
        print("decoder pos_enc after interpolate", pos_enc.size())
        assert int(h0) == pos_enc.shape[-2] and int(w0) == pos_enc.shape[-1]
        pos_enc = pos_enc.permute(0, 2, 3, 4, 1).view(1, new_patches_per_dim, -1, dim).view(1, -1, dim)
        # print("pos decoder embed  after interpolate", pos_enc.size())
        
        state_dict["decoder_pos_embed"] = pos_enc


    return state_dict
    
def interpolate_pretrained_pos_enc_encoder(args: dict, state_dict: dict, seg_temporal_pos=False) -> dict:
    """
    Adjusts the pretrained positional encoding tensor to fit the current model's dimensions.(larger)

    Args:
        args (dict): The input arguments to the model
        state_dict (dict): The loaded state dictionary to adjust

    Returns:
        dict: The adjusted state dictionary with the updated positional encoding
    """
    orig_patches_per_dim = 224 // 16  # original 224x224 model with patch size 16
    new_patches_per_dim = args.img_size // 16
    if orig_patches_per_dim != new_patches_per_dim:
        if not seg_temporal_pos:
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            h0, w0 = new_patches_per_dim + 0.1, new_patches_per_dim + 0.1
            # print("pos_enc before interpolate",  state_dict["pos_embed_spatial"].size()) # ([1, 196, 1024])
            pos_enc = state_dict["pos_embed_spatial"].reshape(
                1, orig_patches_per_dim, orig_patches_per_dim, -1
            )
            print("pos_enc before interpolate", pos_enc.size())
            dim = pos_enc.shape[-1]
            pos_enc = pos_enc.permute(0, 3, 1, 2)
            pos_enc = torch.nn.functional.interpolate(
                pos_enc,
                scale_factor=(h0 / orig_patches_per_dim, w0 / orig_patches_per_dim),
                mode="bicubic",
                align_corners=False,
            )
            assert int(h0) == pos_enc.shape[-2] and int(w0) == pos_enc.shape[-1]
            pos_enc = pos_enc.permute(0, 2, 3, 1).view(1, -1, dim)
            print("pos_enc after interpolate", pos_enc.size())
            state_dict["pos_embed_spatial"] = pos_enc
        else:
            raise NotImplementedError

    # check pos_embed_temporal
    orig_pos_embed_temporal_dim = 8
    new_pos_embed_temporal_dim = args.num_frames // args.t_patch_size
    if orig_pos_embed_temporal_dim != new_pos_embed_temporal_dim:
        pos_enc = state_dict["pos_embed_temporal"].reshape(
                1, orig_pos_embed_temporal_dim, -1
            )
        print("pos_enc temporal before interpolate", pos_enc.size())
        dim = pos_enc.shape[-1]
        pos_enc = pos_enc.permute(0, 2, 1)
        pos_enc = torch.nn.functional.interpolate(
            pos_enc,
            size=(new_pos_embed_temporal_dim,),
            mode="linear",
            align_corners=False,
        )
        assert new_pos_embed_temporal_dim == pos_enc.shape[-1], pos_enc.shape
        pos_enc = pos_enc.permute(0, 2, 1).view(1, -1, dim)
        print("pos_enc temporal after interpolate", pos_enc.size())
        state_dict["pos_embed_temporal"] = pos_enc


    return state_dict


def adjust_state_dict_keys(state_dict: dict) -> dict:
    """
    Adjust the keys of the state dict to match the model.

    Args:
        state_dict (dict): The state dict to adjust

    Returns:
        dict: The adjusted state dict
    """
    if "pred_head.transforms.0.4.weight" not in state_dict:
        return state_dict
    adjusted_state_dict = {}
    adjusted_state_dict["decoder_norm.weight"] = state_dict.pop(
        "pred_head.transforms.0.4.weight"
    )
    adjusted_state_dict["decoder_norm.bias"] = state_dict.pop(
        "pred_head.transforms.0.4.bias"
    )
    # if args.model.pred_t_dim == 8:
    #     adjusted_state_dict["decoder_pred.weight"] = state_dict.pop(
    #         "pred_head.projections.0.weight"
    #     )
    #     adjusted_state_dict["decoder_pred.bias"] = state_dict.pop(
    #         "pred_head.projections.0.bias"
    #     )
        
    for key in state_dict.keys():
        adjusted_state_dict[
            key.replace("pred_head.transforms.0", "decoder_blocks")
        ] = state_dict[key]


    return adjusted_state_dict




def load_mamba_state_dict(model, cfg, prefix='', ignore_missing="relative_position_index", optimizer=None, loss_scaler=None):
    """
    load mamba model for finetune
    """
    checkpoint = torch.load(cfg.pretrained_weights, map_location='cpu')
    print("Load mamba ckpt from %s" % cfg.pretrained_weights)
    checkpoint_model = None
    model_key = 'model|module'
    if cfg.resume:
        cfg.start_epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        

    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint     

    # print("checkpoint_model", checkpoint_model.keys())
    if 'clip_decoder.0.head.weight' in checkpoint_model.keys():
        del checkpoint_model['clip_decoder.0.head.weight']
        del checkpoint_model['clip_decoder.0.head.bias']
        del checkpoint_model['clip_decoder.0.norm.weight']
        del checkpoint_model['clip_decoder.0.norm.bias']

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
    num_patches = model.patch_embed.num_patches # 
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)       
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> B, H, W, C -> B, C, H, W
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # B, C, H, W -> B, H, W, C ->  B, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size, new_size, embedding_size) 
        pos_tokens = pos_tokens.flatten(1, 2) # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

    # we use 8 frames for pretraining
    temporal_pos_embed = checkpoint_model['temporal_pos_embedding']
    orig_t_size = 8 // model.kernel_size
    new_t_size = model.num_frames // model.kernel_size

    if orig_t_size != new_t_size:
        print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        temporal_pos_embed = torch.nn.functional.interpolate(
            temporal_pos_embed, size=(new_t_size,), mode='linear', align_corners=False
        )
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        checkpoint_model['temporal_pos_embedding'] = temporal_pos_embed


    state_dict = checkpoint_model
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def load_pretrained_weights(network, fname, configuration=None, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname)

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod


    if 'vit' in configuration or 'hiera' in configuration or 'mae' in configuration:
        use_nnunet = False
        # decide what the model it is
        if 'mamba' in configuration:
            cfg = omegaconf.OmegaConf.create(configuration)
            pretrained_dict = saved_model
            cfg.pretrained_weights = fname
            cfg.resume = False
            load_mamba_state_dict(mod, cfg, prefix='', optimizer=None, loss_scaler=None)
            # quit this function
            return

        if 'vit' in configuration or 'hiera' in configuration or 'mae' in configuration:
            if 'model_state' in saved_model.keys():
                pretrained_dict = saved_model['model_state']
                nnunet_mode = False
            elif 'model' in saved_model.keys():
                pretrained_dict = saved_model['model']
                nnunet_mode = False
            elif 'network_weights' in saved_model.keys():
                pretrained_dict = saved_model['network_weights']
                nnunet_mode= True
            else:
                raise ValueError("Could not find the model state in the loaded model")
            # if is mae or hiera encoder 
            if not nnunet_mode: 
                pretrained_dict = adjust_state_dict_keys(pretrained_dict)

                pretrained_dict["decoder_pos_embed"] = pretrained_dict["decoder_pos_embed"][:, 1:, :]
                # check if we need to interpoalte the positional encoding
                # input size 
                if 'reshape' in configuration:
                    args = {'img_size': 112, 'num_frames': 64, 't_patch_size': 2}
                    args = omegaconf.OmegaConf.create(args)
                    pretrained_dict = interpolate_pretrained_pos_enc_encoder(args, pretrained_dict)
                    pretrained_dict = interpolate_pretrained_pos_enc_decoder(args, pretrained_dict)

    else:
        use_nnunet = True
        pretrained_dict = saved_model['network_weights']


    model_dict = mod.state_dict()

    if 'pad_seg_layers' in configuration:
        skip_strings_in_pretrained = []
        # change the weights and bias of seg_layers
        for key, _ in model_dict.items():
            if all([i in key for i in skip_strings_in_pretrained]):   
                new_model_shape = model_dict[key].shape
                # assert new_model_shape equal to pretrained_dict[key].shape except for first channel
                assert new_model_shape[1:] == pretrained_dict[key].shape[1:], \
                    f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                    f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                    f"does not seem to be compatible with your network."
                to_pad_weight = pretrained_dict[key]
                target_shape =  new_model_shape
                # pad the weight
                # print("original shape", to_pad_weight.shape)
                # print("target_shape", target_shape)
                # pad only if target shape is not empty list
                if len(target_shape) > 0:
                    print("padding weight for key", key)
                    pad_weight = torch.zeros(target_shape)
                    pad_weight[:to_pad_weight.shape[0],...] = to_pad_weight
                    pretrained_dict[key] = pad_weight

    else:
        skip_strings_in_pretrained = [
            '.seg_layers.',
        ]
    # skip seg_layers means last layer is always reinitialized

    # verify that all but the segmentation layers have the same shape
    if use_nnunet:
        for key, _ in model_dict.items():
            if all([i not in key for i in skip_strings_in_pretrained]):
                import json
                assert key in pretrained_dict, \
                    f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                    f"compatible with your network."
                
                assert model_dict[key].shape == pretrained_dict[key].shape, \
                    f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                    f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                    f"does not seem to be compatible with your network."

        # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
        # encoders. Not supported by this function though (see assertions above)

        # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
        # pretrained_dict = {'module.' + k if is_ddp else k: v
        #                    for k, v in pretrained_dict.items()
        #                    if (('module.' + k if is_ddp else k) in model_dict) and
        #                    all([i not in k for i in skip_strings_in_pretrained])}

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

        model_dict.update(pretrained_dict)
    else:
        if 'scratch_decoder' in configuration:
            print("################### do not load decoder weight  ###################")
            # remove all keys that have 'decoder' in them
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'decoder' not in k}
            # print("pretrained_dict", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        
    # mod.load_state_dict(model_dict)
    missing, unexpected = mod.load_state_dict(
        model_dict, strict=False
    )

    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    print("################### Done ###################")
