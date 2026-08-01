import os
import json
import wandb
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from thop import profile

import configs
# from model.network import GFNet
# from model.network_new import GFNet
# from model.network_new_RepLKinALL import GFNet
from model.network_new_GRL import GFNet
from datasets.homography_dataset_large_size import HomographyDataset, RandomGaussianBlur
from losses.robust_loss import RobustLosses
from benchmark import MultimodalHomogBenchmark
from trainer.train import train_k_steps_cosine
from checkpointing import CheckPoint

def train(args):
    use_ddp = False
    if use_ddp:
        dist.init_process_group('nccl')
        gpus = int(os.environ['WORLD_SIZE'])
        # create model and move it to GPU with id rank
        rank = dist.get_rank()
        print(f"Start running DDP on rank {rank}")
    else:
        # --- 单 GPU 模式 ---
        gpus = 1
        rank = 0  # 唯一进程
        print("Running in Single-GPU Mode")

    device_id = rank % torch.cuda.device_count()
    configs.cfg.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)

    # --- wandb 设置 ---
    # 只有主进程 (rank 0) 开启 wandb
    wandb_log = not args.dont_log_wandb
    experiment_name = args.dataset
    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="GFNet_small", name=experiment_name, reinit=False, mode=wandb_mode)
    checkpoint_dir = "workspace/"

    with open(args.conf_path, 'r') as file:
        conf = json.load(file)   
    training_resolution = (448, 448)
    upsampling_resolution = (560, 560)

    use_small = args.use_small
    use_fm_head = args.use_fm_head
    
    model = GFNet(conf=conf,
                  initial_res=training_resolution,
                  upsample_res=upsampling_resolution,
                  amp=True,
                  symmetric=True,
                  upsample_preds=True,
                  attenuate_cert=True,
                  use_small=use_small,
                  use_fm_head=use_fm_head).cuda()
    print(f'initial_res: {model.initial_res}\n')
    print(f'upsample_res: {model.upsample_res}\n')
    print(f'symmetric: {model.symmetric}\n')
    print(f'upsample_preds: {model.upsample_preds}\n')
    print(f'attenuate_cert: {model.attenuate_cert}\n')
    # batch = {"im_A": torch.randn(1, 3, 448, 448).cuda(),
    #          "im_B": torch.randn(1, 3, 448, 448).cuda(),
    #          }
    # total_ops, total_params = profile(model, inputs=(batch,))
    # print(f"Total MACs: {total_ops / 1e6:.2f} M")
    # print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    
    if args.ft:
        states = torch.load(args.ft_ckpt, map_location='cpu')
        model.load_state_dict(states["model"])
        print(f'Basic model loaded at path:{args.ft_ckpt}. Fine-tuning...')
    if use_ddp:
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        ddp_model = model
    
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    configs.cfg.STEP_SIZE = step_size
    
    N = 2_000_000  # 2M pairs
    # N = 1000  # 2M pairs
    # checkpoint every
    k = 25000 // configs.cfg.STEP_SIZE
    # k = 500 // configs.cfg.STEP_SIZE
    total_epochs = N // ( k * configs.cfg.STEP_SIZE)
    print(f'k: {k}, total_epochs: {total_epochs}')

    if 'glunet' not in args.dataset:
        train_dataset = HomographyDataset(dataset=args.dataset,
                                            mode='train',
                                            input_resolution=(448, 448),
                                            initial_transforms =transforms.Compose([
                                                    transforms.Resize(size=640, antialias=None),
                                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                                    RandomGaussianBlur(p=0.5),
                                                    transforms.ToTensor()
                                                    ]),
                                            bi=True,
                                            normalize=True,
                                            deformation_ratio=[0.3],
                                            )
    else:
        train_dataset = HomographyDataset(dataset=args.dataset,
                                            mode='train',
                                            input_resolution=(448, 448),
                                            initial_transforms =transforms.Compose([
                                                        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
                                                        transforms.RandomGrayscale(p=0.2),
                                                        RandomGaussianBlur(p=0.5),
                                                        transforms.ToTensor()
                                                        ]),
                                            bi=True,
                                            normalize=True)
    print(f'train dataset number: {len(train_dataset)}')
    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={1:4, 2:4, 4:8, 8:8},
        local_largest_scale=8,
        depth_interpolation_mode="bilinear",
        alpha = 0.5,
        c = 1e-4,
        iteration_base=1,
        )
    parameters = [
        {"params": model.parameters(), "lr": configs.cfg.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    megadense_benchmark = MultimodalHomogBenchmark(dataset=args.dataset, input_resolution=448)
    
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    configs.cfg.GLOBAL_STEP = global_step
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
    grad_clip_norm = 0.01

    try:
        for n in range(configs.cfg.GLOBAL_STEP, N, k * configs.cfg.STEP_SIZE):
            mega_sampler = torch.utils.data.RandomSampler(
                train_dataset, num_samples = batch_size * k, replacement=False
            )
            mega_dataloader = iter(
                torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size = batch_size,
                    sampler = mega_sampler,
                    num_workers = 8,
                )
            )
            train_k_steps_cosine(
                n, k, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
            )
            
            checkpointer.save(model, optimizer, lr_scheduler, configs.cfg.GLOBAL_STEP)
   
        ## save the final ckpt
        checkpointer.save(model, optimizer, lr_scheduler, configs.cfg.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step = configs.cfg.GLOBAL_STEP)
    except KeyboardInterrupt:
        if not args.dont_log_wandb:
            checkpointer.save(model, optimizer, lr_scheduler, configs.cfg.GLOBAL_STEP)
        sys.exit(0)
    
if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    
    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--gpu_batch_size", default=8, type=int)
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--ft", action='store_true', default=False)
    parser.add_argument("--ft_ckpt", type=str, default='./ckpts/basic/last.pth')
    parser.add_argument("--use_small", action='store_true', default=False)
    parser.add_argument("--use_fm_head", action='store_true', default=False)
    args, _ = parser.parse_known_args()

    train(args)