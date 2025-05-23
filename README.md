<h1 align="center"> FourierFlow: <br>Frequency-aware Flow Matching for Generative Turbulence Modeling
</h1>


### 1. Environment setup

```bash
conda create -n repa python=3.9 -y
conda activate repa
pip install -r requirements.txt
```

All our experiments are implemented on 8 x NVIDIA H800 GPUs.

### 2. Dataset

#### Dataset download

Please download Compressible N-S datasets in PDEBench and Shear Flow in The Well datasets, and set 'base-path' in `train.py` accordingly.
([PDEBench](https://github.com/pdebench/PDEBench), [The Well](https://github.com/PolymathicAI/the_well))


### 3. Training

<!-- ```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_ablation.py
``` -->
```bash
accelerate launch /you_path_to/FourierFlow/train.py --allow-tf32
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--proj-coeff`: Any values larger than 0
- `--encoder-depth`: Any values between 1 to the depth of the model
- `--output-dir`: Any directory that you want to save checkpoints and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)


### 4. Evaluation

```bash
python eval.py
```

If you want to test the surrogate model and the generative model together, run the following:

```bash
python all_eval.py
```

If you want to test the results of each step, run:

```bash
python all_eval_step.py
```


We also provide the SiT-XL/2 checkpoint (trained for 4M iterations) used in the final evaluation. It will be automatically downloaded if you do not specify `--ckpt`.


### Analysis

1. analysis the power of diffTrans
```bash
python diff_analysis.py
```


## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), and [SiT](https://github.com/willisma/SiT) repositories.


