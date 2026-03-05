# telescope_gym Examples

CleanRL-based PPO training scripts demonstrating two telescope adaptive optics control problems.

## Setup

Install telescope_gym with example dependencies:

```bash
pip install -e ".[examples]"
```

## Examples

### Atmospheric PTT Correction (`atmos_ptt_ppo.py`)

Train a CNN policy to correct atmospheric wavefront errors on a 15-aperture telescope using piston-tip-tilt control.

```bash
python atmos_ptt_ppo.py
python atmos_ptt_ppo.py --total-timesteps 5000000 --num-envs 8
python atmos_ptt_ppo.py --track  # Enable W&B logging
```

**Environment setup:**
- Observation: Single focal-plane PSF image (1, 128, 128)
- Action: 45 continuous values (15 apertures x 3 PTT)
- Reward: Strehl ratio improvement
- Episode: 30 steps with atmospheric turbulence evolving at 200Hz

### Dark Hole Digging (`dark_hole_ppo.py`)

Train a CNN policy to suppress speckles in a dark hole region using piston-only control.

```bash
# First generate random initial states
python dark_hole_ppo.py --generate-init-states

# Then train
python dark_hole_ppo.py --init-states-pkl init_states.pkl
```

**Environment setup:**
- Observation: Single focal-plane PSF image (1, 128, 128)
- Action: 15 continuous values (piston-only, tip-tilt masked)
- Reward: Dark hole contrast improvement
- Episode: 200 steps max

## Model Architecture

Both examples use the CNN actor-critic from `models/focal_plane_cnn.py`:
- 3 convolutional layers (128 channels, 5x5 kernel, stride 2)
- Shared CNN trunk for actor and critic
- 3 dense layers (128 units) for each head
- Separate action mean and log-std outputs

## Monitoring

Training logs are written to `runs/` via TensorBoard:

```bash
tensorboard --logdir runs/
```
