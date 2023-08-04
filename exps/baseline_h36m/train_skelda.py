import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import tqdm
from config import config
from model import siMLPe as Model
from test_skelda import test
from torch.utils.tensorboard import SummaryWriter
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import ensure_dir, link_file

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

# datamode = "gt-gt"
datamode = "pred-pred"

dconfig = {
    "item_step": 1,
    "window_step": 1,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_10fps.json"
# ]

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_kppspose_10fps.json",
]

# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"

tconfig = dict(dconfig)

# ==================================================================================================


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-name", type=str, default=None, help="=exp name")
parser.add_argument("--seed", type=int, default=888, help="=seed")
parser.add_argument("--temporal-only", action="store_true", help="=temporal only")
parser.add_argument(
    "--layer-norm-axis", type=str, default="spatial", help="=layernorm axis"
)
parser.add_argument("--with-normalization", action="store_true", help="=use layernorm")
parser.add_argument("--spatial-fc", action="store_true", help="=use only spatial fc")
parser.add_argument("--num", type=int, default=64, help="=num of blocks")
parser.add_argument("--weight", type=float, default=1.0, help="=loss weight")
parser.add_argument(
    "--ckptdir", type=str, default="./log/snapshot/", help="=checkpoint directory"
)
parser.add_argument(
    "--model_weights_path",
    type=str,
    default="",
    help="directory with the model weights to copy",
)

args = parser.parse_args()

torch.use_deterministic_algorithms(False)
acc_log = open(args.exp_name, "a")
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num
config.snapshot_dir = os.path.abspath(args.ckptdir)

acc_log.write("".join("Seed : " + str(args.seed) + "\n"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)


dconfig["input_n"] = config.motion.h36m_input_length
dconfig["output_n"] = config.motion.h36m_target_length_eval
tconfig["input_n"] = config.motion.h36m_input_length
tconfig["output_n"] = config.motion.h36m_target_length_eval

# ==================================================================================================


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

# ==================================================================================================


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


# ==================================================================================================


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


# ==================================================================================================


def train_step(
    h36m_motion_input,
    h36m_motion_target,
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
):

    if config.deriv_input:
        b, n, c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(
            dct_m[:, :, : config.motion.h36m_input_length], h36m_motion_input_.cuda()
        )
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.cuda())
    motion_pred = torch.matmul(
        idct_m[:, : config.motion.h36m_input_length, :], motion_pred
    )

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, : config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, : config.motion.h36m_target_length]

    b, n, c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, config.motion.dim // 3, 3).reshape(-1, 3)
    h36m_motion_target = (
        h36m_motion_target.cuda()
        .reshape(b, n, config.motion.dim // 3, 3)
        .reshape(-1, 3)
    )
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b, n, config.motion.dim // 3, 3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b, n, config.motion.dim // 3, 3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar("Loss/angle", loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer
    )
    writer.add_scalar("LR/train", current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


# ==================================================================================================

stime = time.time()

model = Model(config)
model.train()
model.cuda()

print(
    "Total number of parameters of the network is: "
    + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
)

config.motion.h36m_target_length = config.motion.h36m_target_length_train
eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
tconfig["output_n"] = config.motion.h36m_target_length

# Load preprocessed datasets
print("Loading datasets ...")
dataset_train, dlen_train = [], 0
for dp in datasets_train:
    cfg = copy.deepcopy(dconfig)
    if "mocap" in dp:
        cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"

    ds, dlen = utils_pipeline.load_dataset(dp, "train", cfg)
    dataset_train.extend(ds["sequences"])
    dlen_train += dlen

esplit = "test" if "mocap" in dataset_eval_test else "eval"
cfg = copy.deepcopy(dconfig)
if "mocap" in dataset_eval_test:
    cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
dataset_eval, dlen_eval = utils_pipeline.load_dataset(dataset_eval_test, esplit, cfg)
dataset_eval = dataset_eval["sequences"]

# initialize optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.cos_lr_max, weight_decay=config.weight_decay
)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, "train")
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None:
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

if args.model_weights_path != "":
    print("Loading model weights from:", args.model_weights_path)
    model.load_state_dict(torch.load(args.model_weights_path))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.0
avg_lr = 0.0
best_loss = np.inf
n_epochs = 0

while (nb_iter + 1) < config.cos_lr_total_iters:

    label_gen_train = utils_pipeline.create_labels_generator(dataset_train, tconfig)
    label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, dconfig)
    print("Training epoch {} ...".format(n_epochs))
    n_epochs += 1

    nbatch = config.batch_size
    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
        total=int(dlen_train / nbatch),
    ):

        sequences_train = utils_pipeline.make_input_sequence(batch, "input", datamode)
        sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

        augment = True
        if augment:
            sequences_train, sequences_gt = utils_pipeline.apply_augmentations(
                sequences_train, sequences_gt
            )

        # Merge joints and coordinates to a single dimension
        sequences_train = sequences_train.reshape(
            [nbatch, sequences_train.shape[1], -1]
        )
        sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

        # Convert to meters
        sequences_train = sequences_train / 1000.0
        sequences_gt = sequences_gt / 1000.0

        sequences_train = torch.from_numpy(sequences_train).to(device)
        sequences_gt = torch.from_numpy(sequences_gt).to(device)

        h36m_motion_input = sequences_train
        h36m_motion_target = sequences_gt

        loss, optimizer, current_lr = train_step(
            h36m_motion_input,
            h36m_motion_target,
            model,
            optimizer,
            nb_iter,
            config.cos_lr_total_iters,
            config.cos_lr_max,
            config.cos_lr_min,
        )
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0:
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) == config.cos_lr_total_iters:
            break
        nb_iter += 1

    torch.save(
        model.state_dict(),
        config.snapshot_dir + "/model-iter-" + str(nb_iter + 1) + ".pth",
    )

    model.eval()
    acc_tmp = test(eval_config, model, label_gen_eval, dlen_eval, nbatch, datamode)
    print(acc_tmp)

    if acc_tmp[-1] < best_loss:
        best_loss = acc_tmp[-1]
        torch.save(
            model.state_dict(),
            config.snapshot_dir + "/model-best" + ".pth",
        )
        print("Saved new best model")

    acc_log.write("".join(str(nb_iter + 1) + "\n"))
    line = ""
    for ii in acc_tmp:
        line += str(ii) + " "
    line += "\n"
    acc_log.write("".join(line))
    model.train()

writer.close()

ftime = time.time()
print("Training took {} seconds".format(int(ftime - stime)))
