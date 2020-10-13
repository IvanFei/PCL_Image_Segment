import os
import shutil

import torch


def load_checkpoint(model, optimizer, checkpoint_f):
    if isinstance(checkpoint_f, str):
        print(f"[*] LOADED: checkpoint from: {checkpoint_f}")
        checkpoint = torch.load(checkpoint_f)
        checkpoint = checkpoint["state_dict"]
    else:
        raise TypeError(f"[*] ERROR: checkpoint is str type.")

    checkpoint_dict = {}
    for k, v in checkpoint.items():
        if "num_batches_tracked" in k:
            continue

        if k.startswith("network."):
            checkpoint_dict[k[8:]] = v
        else:
            checkpoint_dict[k] = v

    model.load_state_dict(checkpoint_dict)


def get_save_checkpoint_info(checkpoint_f):

    def get_number(item, delimiter, idx, replace_word, must_contain=""):
        if must_contain in item:
            return int(item.split(delimiter)[idx].replace(replace_word, ""))

    basename = os.path.basename(checkpoint_f.rsplit(".", 1)[0])
    epoch = get_number(basename, "_", 1, "epoch")
    step = get_number(basename, "_", 2, "step", "model")

    return epoch, step


def save_checkpoint(checkpoint_dir, save_name, model):
    if not isinstance(checkpoint_dir, str):
        raise ValueError("[*] ERROR: checkpoint_dir should be str.")

    checkpoint_path = os.path.join(checkpoint_dir, save_name)

    torch.save(model.state_dict(), checkpoint_path)


