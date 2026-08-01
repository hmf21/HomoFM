# HomoFM
[HomoFM: Efficient and Robust Homography Estimation with Conditional Flow-Based Displacement Refinement repo for submission](https://arxiv.org/abs/2601.18222)

# Quick Start
1. Torch version: 2.3.1
```
conda create --name HomoFM python==3.10.13 && \
conda activate HomoFM && \
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. Dataset Download

Please download the dataset given in [GFNet](https://github.com/KN-Zhang/GFNet), we can use the same dataset to evaluate the model.

4. Model test

  For MSCOCO dataset
```
python -m test --dataset mscoco --conf_path configs/basic_small.json --ckpt_path workspace/glunet_448x448_occlusion/latest.pth --use_fm_head
```

  For GoogleMap dataset
```
python -m test --dataset googlemap_448x448 --conf_path configs/map_small.json --ckpt_path workspace/googlemap/latest.pth --use_fm_head
```

  For VIS-IR dataset
```
python -m test --dataset vis_ir_drone --conf_path configs/vis_ir_small.json --ckpt_path workspace/vis_ir_drone/latest.pth --use_small --use_fm_head
```

4. For more training details and weights, please inform us.

# 📚 Citation
If you find this work helpful, please cite our paper:

```
@misc{he2026homofmdeephomographyestimation,
      title={HomoFM: Deep Homography Estimation with Flow Matching}, 
      author={Mengfan He and Liangzheng Sun and Chunyu Li and Ziyang Meng},
      year={2026},
      eprint={2601.18222},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.18222}, 
}
```

# 🙏 Acknowledgement

This project is built upon the [GFNet](https://github.com/KN-Zhang/GFNet) codebase.
We sincerely thank the original authors for their impressive work.
