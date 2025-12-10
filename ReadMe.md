# Knowledge Distillation for Image Classification

## Environment Setup:
```sh
$ conda create -n kd-venv python=3.10
$ conda activate kd-venv
$ pip install -r requirments.txt
```

### Defaults:
**Teacher Model:** ResNet-152 (pretrained on ImageNet) <br>
**Student Model:** MobileNetV2 <br>
**Dataset:** [mini-imagenet](https://huggingface.co/datasets/timm/mini-imagenet)

## Train Student
```sh
$ python train_student.py
```

## Curriculum Learning
Several sampling strategies are implemented in `curriculum_dataloader`. In the adaptive strategy, a fraction (by default 70%) of the dataset is selected around a given difficulty.

### Train Student with Curriculum Learning
```sh
$ python train_student_rl.py
```

### Difficulty Score
Implemented in `models.py`. <br>
Let `p_t(i) = P_teacher(i | x)` and `p_s(i) = P_student(i | x)` for `i = 1..100`.

- Teacher confidence:
  - `D_conf = 1 - max_i p_t(i)`

- Normalized entropy:
  - `D_entropy = ( - sum_{i=1..100} p_t(i) * log(p_t(i)) ) / log(100)`

- Student-teacher disagreement (KL, scaled + clipped):
  - `DKL = sum_{i=1..100} p_t(i) * log( p_t(i) / p_s(i) )`
  - `D_kl = min( DKL / 5, 1 )`

- Combined difficulty:
  - `D_sample = 0.3*D_conf + 0.3*D_entropy + 0.4*D_kl`  (in `[0, 1]`)

### Reward function (for RL agent)
Implemented in `rl_curriculum_env.py`. <br>
Let `a_t` be current validation accuracy (%) and `l_t` be current training loss.  
Let `a_{t-1}` and `l_{t-1}` be the previous values.  
Let `e` be the current epoch index, and `E` be `num_epochs`.

- Validation accuracy improvement (scaled)
If `a_{t-1} > 0`:
    - `R_acc = 2.0 * (a_t - a_{t-1})`
Else:
    - `R_acc = 0`

- Loss stability penalty (only when loss increases, capped)
If `l_{t-1} < +inf`:
    - `Δl = l_t - l_{t-1}`
    - `R_loss = - min( 5.0 * max(Δl, 0), 5.0 )`
Else:
    - `R_loss = 0`

- Early high-accuracy bonus (fast convergence)
    - `R_early = 2.0 * I( a_t > 60.0 AND e < 0.3*E )`

- Absolute performance bonus (high accuracy)
    - `R_abs = 0.1 * max(a_t - 70.0, 0)`

- Total reward
    - `R_t = R_acc + R_loss + R_early + R_abs`
