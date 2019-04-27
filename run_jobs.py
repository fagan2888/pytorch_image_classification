import concurrent.futures
import os
import subprocess
import sys
import uuid

import numpy as np

NUM_GPUS = 3
GPU_INDICES = [0, 1, 2]
max_workers_per_gpu = 1 # num models per GPU

RESULTS_DIR = "/home/john/pytorch_image_classification/results/cifar_search"

assert len(GPU_INDICES) == NUM_GPUS


def sample_command():
    # Always train a depth 110 resnet with preactivation
    base_cmd = "python -u train.py --arch resnet_preact --depth 110".split()
    
    # Potential to seed later
    rng = np.random.RandomState(None)

    def coin_flip():
        return rng.choice([True, False])

    def unif_sample(low, high):
        return (high - low) * rng.rand() + low

    # Architecture
    block_type = rng.choice(["basic", "bottleneck"])
    base_cmd.extend(["--block_type", block_type])

    base_channels = rng.choice([4, 8, 16, 32])
    base_cmd.extend(["--base_channels", str(base_channels)])

    # Remove first relu
    remove_relu = coin_flip()
    base_cmd.extend(["--remove_first_relu", str(remove_relu)])

    use_last_bn = coin_flip()
    base_cmd.extend(["--add_last_bn", str(use_last_bn)])

    preact_stage = str([coin_flip() for _ in range(3)]).lower()
    base_cmd.extend(["--preact_stage", preact_stage])

    # Data augmentation
    ## standard data augmentation
    if coin_flip():
        base_cmd.extend(['--use_random_crop', "True"])
        base_cmd.extend(['--random_crop_padding', str(rng.choice([2, 4, 8]))])
    if coin_flip():
        base_cmd.extend(['--use_horizontal_flip', "True"])

    if coin_flip():
        base_cmd.extend(["--use_cutout"])
        base_cmd.extend(["--cutout_size", str(rng.choice([8, 12, 16]))])
        if coin_flip():
            base_cmd.extend(['--cutout_inside'])
    elif coin_flip():
        base_cmd.extend(["--use_dual_cutout"])
        base_cmd.extend(['--dual_cutout_alpha', str(unif_sample(0.05, 0.3))])
    elif coin_flip():
        base_cmd.extend(["--use_random_erasing"])
        base_cmd.extend(["--random_erasing_prob", str(unif_sample(0.2, 0.8))])
    elif coin_flip():
        base_cmd.extend(["--use_mixup"])
        base_cmd.extend(["--mixup_alpha", str(unif_sample(0.6, 1.4))])
    elif coin_flip():
        base_cmd.extend(["--use_ricap"])
        base_cmd.extend(["--ricap_beta", str(unif_sample(0.1, 0.6))])

    # Optimization
    base_cmd.extend(["--epochs", str(200)])
    base_cmd.extend(["--batch_size", str(rng.choice([32, 64, 128, 256]))])
    base_cmd.extend(["--base_lr", str(rng.choice(np.linspace(start=1e-4, stop=0.5, num=20)))])
    base_cmd.extend(["--weight_decay", str(rng.choice(np.logspace(-5, -1, 10)))])

    if coin_flip():
        base_cmd.extend(["--no_weight_decay_on_bn"])

    # Either SGD or Adam
    if coin_flip():
        base_cmd.extend(["--optimizer", "sgd"])
        base_cmd.extend(["--nesterov", str(coin_flip())])
        base_cmd.extend(["--momentum", str(rng.choice(np.linspace(start=0.6, stop=0.99, num=10)))])
    else:
        base_cmd.extend(["--optimizer", "adam"])
        base_cmd.extend(["--betas", '[0.9, 0.999]'])

    # Choice of learning rate schedule
    if coin_flip():
        base_cmd.extend(["--scheduler", "multistep"])
        step1 = rng.choice([40, 60, 80, 100])
        delta = rng.choice([20, 40, 60])
        milestones = str([step1, step1 + delta])
        base_cmd.extend(["--milestones", milestones])
    elif coin_flip():
        base_cmd.extend(["--scheduler", "cosine"])

    # Misc
    seed = rng.randint(0, 99999)
    base_cmd.extend(["--seed", str(seed)])

    if coin_flip():
        base_cmd.extend(["--use_label_smoothing"])
        base_cmd.extend(["--label_smoothing_epsilon", str(rng.choice(np.linspace(0.01, 0.2, 10)))])

    name = str(uuid.uuid4().hex)
    base_cmd.extend(["--outdir", os.path.join(RESULTS_DIR, name)])

    return base_cmd


def launch_jobs(gpu_idx):
    env = os.environ
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDICES[gpu_idx])
    
    num_fails = 0
    # Endlessly train models
    while True:
        command = sample_command()
        try:
            subprocess.call(command, env=env)
        except:
            print("FAILED")
            print(" ".join(command))
            num_fails += 1
            if num_fails > 5:
                raise


with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GPUS * max_workers_per_gpu) as executor:
    for idx in range(NUM_GPUS * max_workers_per_gpu):
        executor.submit(launch_jobs, idx % NUM_GPUS)
