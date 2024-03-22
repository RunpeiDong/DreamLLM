import re


# TODO: Make it more universal.
def vit_lr_scale_func_llrd(key):
    if "clip_vision_embedding.clip_vision_model.encoder.layers" in key:
        in_pp_layer = int(re.findall(f"layers\.(\d+)\.", key)[0])
        decay = 0.9 ** (23 - in_pp_layer - 1)
        # decay = 0.81 ** (23 - in_pp_layer - 1)
        # decay = 0.93 ** (32 - in_pp_layer - 1)
        return decay
    elif "clip_vision_model" in key:
        # return 0.01
        return 0.1  # a smaller learning rate for vision encoder
    return 1


def vit_lr_scale_func(key):
    if "clip_vision_model" in key:
        # return 0.01
        return 0.1  # a smaller learning rate for vision encoder
    return 1


def llm_lr_scale_func(key):
    if "model.layers" in key:
        in_pp_layer = int(re.findall(f"layers\.(\d+)\.", key)[0])
        decay = 0.931 ** (32 - in_pp_layer - 1)
        return decay
    return 1


def get_param_groups(model, no_weight_decay_cond, scale_lr_cond, lr, wd):
    """creates param groups based on weight decay condition (regularized vs non regularized)
    and learning rate scale condition (args.lr vs lr_mult * args.lr)
    scale_lr_cond is used during finetuning where head of the network requires a scaled
    version of the base learning rate.
    """
    wd_no_scale_lr = []
    wd_scale_lr = {}
    no_wd_no_scale_lr = []
    no_wd_scale_lr = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if no_weight_decay_cond is not None:
            no_wd = no_weight_decay_cond(name, param)
        else:
            # do not regularize biases nor Norm parameters
            no_wd = name.endswith(".bias") or len(param.shape) == 1

        if scale_lr_cond is not None:
            lr_mult = scale_lr_cond(name)
            scale_lr = lr_mult != 1
        else:
            scale_lr = False

        if not no_wd and not scale_lr:
            wd_no_scale_lr.append(param)
        elif not no_wd and scale_lr:
            if lr_mult not in wd_scale_lr:
                wd_scale_lr[lr_mult] = [param]
            else:
                wd_scale_lr[lr_mult].append(param)
        elif no_wd and not scale_lr:
            no_wd_no_scale_lr.append(param)
        else:
            if lr_mult not in no_wd_scale_lr:
                no_wd_scale_lr[lr_mult] = [param]
            else:
                no_wd_scale_lr[lr_mult].append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({"params": wd_no_scale_lr, "weight_decay": wd, "lr": lr})
    if len(wd_scale_lr):
        for lr_mult, params in wd_scale_lr.items():
            param_groups.append({"params": params, "weight_decay": wd, "lr": lr * lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({"params": no_wd_no_scale_lr, "weight_decay": 0.0, "lr": lr})
    if len(no_wd_scale_lr):
        for lr_mult, params in no_wd_scale_lr.items():
            param_groups.append({"params": params, "weight_decay": 0.0, "lr": lr * lr_mult})

    return param_groups
