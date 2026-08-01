import os
import json
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm

import configs
# from model.network import GFNet
# from model.network_new import GFNet
from model.network_new_GRL import GFNet
from estimation import demo_estimation, auc

from thop import profile

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--use_small", action='store_true', default=False)
    parser.add_argument("--use_fm_head", action='store_true', default=False)
    args, _ = parser.parse_known_args()

    with open(args.conf_path, 'r') as file:
        conf = json.load(file)   
    training_resolution = (448, 448)
    upsampling_resolution = (560, 560)
    
    model = GFNet(conf=conf,
                  initial_res=training_resolution,
                  upsample_res=upsampling_resolution,
                  symmetric=True,
                  upsample_preds=True,
                  attenuate_cert=True,
                  use_small=args.use_small,
                  use_fm_head=args.use_fm_head).cuda()
    print(f'initial_res: {model.initial_res}\n')
    print(f'upsample_res: {model.upsample_res}\n')
    print(f'symmetric: {model.symmetric}\n')
    print(f'upsample_preds: {model.upsample_preds}\n')
    print(f'attenuate_cert: {model.attenuate_cert}\n')
    batch = {"im_A": torch.randn(1, 3, 448, 448).cuda(),
             "im_B": torch.randn(1, 3, 448, 448).cuda(),
             }
    total_ops, total_params = profile(model, inputs=(batch,))
    print(f"Total MACs: {total_ops / 1e6:.2f} M")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    
    states = torch.load(args.ckpt_path)
    model.load_state_dict(states["model"])

    # my datasets
    if args.dataset == 'mscoco':
        test_path = f'{configs.cfg.DATA_PATH}/test/mscoco_1k_448x448/source'
        ext = 'png'
    elif args.dataset == 'vis_ir_drone':
        test_path = f'{configs.cfg.DATA_PATH}/test/visir_1k_448x448/source'
        ext = 'png'
    elif args.dataset == 'googlemap_448x448':
        test_path = f'{configs.cfg.DATA_PATH}/test/googlemap_1k_448x448_new/source'
        ext = 'jpg'
    elif args.dataset == 'googlemap_224x224':
        test_path = f'{configs.cfg.DATA_PATH}/test/googlemap_1k_224x224_new/source'        
        ext = 'jpg'
    elif args.dataset == 'googlemap_672x672':
        test_path = f'{configs.cfg.DATA_PATH}/test/googlemap_1k_672x672/source'        
        ext = 'jpg'
        
    error = []
    runtime = []
    results = {}
    test_list = os.listdir(test_path)
    for image_name in tqdm(test_list):
        img1_path = f'{test_path}/' + image_name
        img2_path = test_path.replace('source', 'target/') + image_name
        H_s2t_path = test_path.replace('source', 'H_s2t/') + image_name.replace(ext, 'json')
    
        output_error, output_time = demo_estimation(model, img1_path, img2_path, H_s2t_path, if_print=False)
        error.append(output_error)
        runtime.append(output_time)
    
    thresholds = [3, 5, 10, 20]
    aucs = auc(error, thresholds)
    results.update({f'auc@{t}_{args.dataset}': v for t, v in zip(thresholds, aucs)})    
    print(results)
    print(f'ACE: {np.mean(error)}')
    print(f'Time: {np.mean(runtime)}')
    

    