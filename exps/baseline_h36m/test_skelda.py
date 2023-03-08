import argparse
import copy
import sys
import time

import numpy as np
import torch
import tqdm
from config import config
from matplotlib import pyplot as plt
from model import siMLPe as Model

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

datamode = "pred-gt"

dconfig = {
    "item_step": 1,
    "window_step": 1,
    # "window_step": 5,
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

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
dataset_eval_test = dataset_eval_test.format("test")

viz_action = -1
# viz_action = 14

# ==================================================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device, dmode):
    sequences = utils_pipeline.make_input_sequence(batch, split, dmode)

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    # Convert to meters
    sequences = sequences / 1000.0

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def viz_joints_3d(sequences_predict, batch):
    batch = batch[0]
    last_input_pose = batch["input"][-1]["bodies3D"][0]

    vis_seq_pred = (
        sequences_predict.cpu()
        .detach()
        .numpy()
        .reshape(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
    )[0]
    utils_pipeline.visualize_pose_trajectories(
        np.array([cs["bodies3D"][0] for cs in batch["input"]]),
        np.array([cs["bodies3D"][0] for cs in batch["target"]]),
        utils_pipeline.make_absolute_with_last_input(vis_seq_pred, last_input_pose),
        batch["joints"],
        # {"room_size": [3200, 4800, 2000], "room_center": [0, 0, 1000]},
        {},
    )

    plt.grid(False)
    plt.axis("off")
    plt.show()


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


def regress_pred(
    model,
    label_gen,
    num_samples,
    joint_used_xyz,
    m_p3d_h36,
    dlen: int,
    nbatch: int,
    dmode: str,
):
    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(label_gen, batch_size=nbatch),
        total=int(dlen / nbatch),
    ):

        if viz_action != -1 and viz_action != batch[0]["action"]:
            continue

        sequences_train = prepare_sequences(batch, nbatch, "input", device, dmode)
        sequences_gt = prepare_sequences(batch, nbatch, "target", device, dmode)

        motion_input = sequences_train
        motion_target = sequences_gt

        motion_input = motion_input.cuda()
        b, n, _ = motion_input.shape
        num_samples += b

        outputs = []
        step = config.motion.h36m_target_length_train
        estep = config.motion.h36m_target_length_eval
        if step == estep:
            num_step = 1
        else:
            num_step = estep // step + 1

        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(
                        dct_m[:, :, : config.motion.h36m_input_length],
                        motion_input_.cuda(),
                    )
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(
                    idct_m[:, : config.motion.h36m_input_length, :], output
                )[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            output = output.reshape(-1, config.motion.dim // 3 * 3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:, :estep]

        if viz_action != -1:
            viz_joints_3d(motion_pred * 1000, batch)

        motion_target = motion_target.detach()
        b, n, _ = motion_target.shape

        motion_gt = motion_target.detach().cpu()
        motion_gt = motion_gt.clone().reshape(b, n, config.motion.dim // 3, 3)

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.clone().reshape(b, n, config.motion.dim // 3, 3)

        mpjpe_p3d_h36 = torch.sum(
            torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim=3), dim=2),
            dim=0,
        )
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()

    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36


# ==================================================================================================


def test(config, model, dataloader, dlen, nbatch, dmode):

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array(
        [
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
        ]
    ).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(
        model, pbar, num_samples, joint_used_xyz, m_p3d_h36, dlen, nbatch, dmode
    )

    return m_p3d_h36


# ==================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model-pth", type=str, default=None, help="=encoder path")
    args = parser.parse_args()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval

    model = Model(config)
    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    dconfig["input_n"] = config.motion.h36m_input_length
    dconfig["output_n"] = config.motion.h36m_target_length_eval

    # Load preprocessed datasets
    cfg = copy.deepcopy(dconfig)
    if "mocap" in dataset_eval_test:
        cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
    dataset_test, dlen = utils_pipeline.load_dataset(dataset_eval_test, "test", cfg)
    dataset_test = dataset_test["sequences"]
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, dconfig)
    dataloader = label_gen_test

    batch_size = 1
    stime = time.time()
    print(test(config, model, dataloader, dlen, batch_size, datamode))

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))
