
# Max-Matching 

AAAA'21: Learning with Group Noise (Pytorch implementation).

========

This is the code for the paper:

[Learning with Group Noise](https://arxiv.org/abs/2103.09468)

Qizhou Wang, Jiangchao Yao, Chen Gong, Tongliang Liu, Mingming Gong, Hongxia Yang, Bo Han. 

To be presented at [AAAI 2021](https://aaai.org/Conferences/AAAI-21/).

  

If you find this code useful in your research then please cite

```bash

@inproceedings{wang2021maxmatching,
title={Learning with Group Noise},
author={Qizhou Wang and Jiangchao Yao and Chen Gong and Tongliang Liu and Mingming Gong and Hongxia Yang and Bo Han},
booktitle={AAAI},
pages={10192--10200},
year={2021}
}

```

  

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA GTX3090 GPU. The environment is as bellow:

  

- Window 10

- CUDA 10.2.89

- Python 3.7.6 (Anaconda 4.9.2 64 bit)

- PyTorch 1.5.0

- numpy 1.18.1

  

## Running Max-Matching on benchmark datasets from Amazon

Here is an example:

  

```bash

python main.py --dataset=Video

```

  

## Performance

  


```markdown
| Dataset  | Video | Video | Beauty | Beauty | Game  | Game  |
|----------|-------|-------|--------|--------|-------|-------|
| Metric   | HIT   | NDCG  | HIT    | NDCG   | HIT   | NDCG  |
| Accuracy | 0.694 | 0.473 | 0.561  | 0.389  | 0.518 | 0.345 |
```

We will release the improved realizations and other applications in the upcoming journal verison. 

Contact: Qizhou Wang (qizhouwang.nanjing@gmail.com); Bo Han (bhanml@comp.hkbu.edu.hk).

  

