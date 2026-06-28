# CoinRun world model research notes

Context: functional interactive world model for Procgen CoinRun, trained from recorded `.npz`
gameplay at 64x64 RGB, with an interactive target of at least 10 FPS.

## Recommended path

Build a small offline, action-conditioned video world model first:

1. Convert every recorded `.npz` episode into contiguous sequences:
   - `obs`: `uint8`, shape `[T, 64, 64, 3]`
   - `actions`: `int64`, shape `[T]`, Procgen discrete action id in `[0, 14]`
   - optionally `rewards`, `dones`
2. Train a compact encoder-decoder:
   - fastest baseline: convolutional autoencoder or VQ-VAE
   - target latent: 8x8 tokens or a 128-512 dimensional vector
3. Train an action-conditioned dynamics model:
   - baseline: GRU/ConvLSTM over latents, predicts next latent or next frame
   - stronger: IRIS-style autoregressive Transformer over discrete VQ tokens plus actions
4. Build an interactive loop:
   - keep the latest latent state in memory
   - read keyboard action
   - predict next latent/frame
   - render at 64x64 scaled up to a window
   - measure FPS and latency

This is the lowest-risk route for the class deliverable because it directly uses the existing
offline `.npz` recordings and focuses on real-time controllable prediction rather than learning a
new policy. A simple ConvLSTM baseline can be implemented quickly; IRIS-style tokenization is the
most promising upgrade if the baseline blurs.

## Candidate implementations

### IRIS

- Repo: https://github.com/eloialonso/iris
- Paper: https://arxiv.org/abs/2209.00588
- Why it matters: discrete autoencoder + autoregressive Transformer world model. The repo includes
  a live mode for keyboard-controlled world-model unrolls (`play.sh -w`), which is very close to
  our required demo.
- Adaptation cost: medium. It targets Atari, so we would replace the environment/dataset loader
  with Procgen `.npz` episodes and adjust action-space assumptions.
- License note: GPL-3.0, so reuse carefully if this project has different licensing constraints.

### DreamerV3

- Repo: https://github.com/danijar/dreamerv3
- Paper: https://arxiv.org/abs/2301.04104
- Why it matters: robust modern latent world model that learns from pixels and predicts future
  representations/rewards conditioned on actions.
- Adaptation cost: medium-high. Excellent if we want policy learning, but more engineering than
  needed for an offline interactive simulator. It expects live environment integration or replay
  plumbing.

### STORM

- Repo: https://github.com/weipu-zhang/STORM
- Paper: https://arxiv.org/abs/2310.09615
- Why it matters: efficient stochastic Transformer world model, also visual RL oriented.
- Adaptation cost: medium-high. The original repo notes it is no longer maintained and recommends
  a newer OC-STORM repo, so treat it more as architecture guidance than a base to fork.

### Classic World Models

- Repo: https://github.com/AppliedDataSciencePartners/WorldModels
- Paper: https://arxiv.org/abs/1803.10122
- Why it matters: simple VAE + recurrent dynamics + controller decomposition, easy to understand
  and explain.
- Adaptation cost: low-medium. Older TensorFlow/Keras style, but the architecture is a good backup
  if we implement our own PyTorch version.

### Procgen reference

- Repo: https://github.com/openai/procgen
- Paper: https://arxiv.org/abs/1912.01588
- Notes: observations are already 64x64 RGB; action space is `Discrete(15)`; interactive human
  stepping is expected around 15 Hz. This lines up well with the required 10 FPS target.

## Papers to cite in the report

- Ha and Schmidhuber, "World Models" / "Recurrent World Models Facilitate Policy Evolution"
  https://arxiv.org/abs/1803.10122 and https://arxiv.org/abs/1809.01999
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet)
  https://arxiv.org/abs/1811.04551
- Hafner et al., "Dream to Control" / Dreamer
  https://arxiv.org/abs/1912.01603
- Hafner et al., "Mastering Diverse Domains through World Models" / DreamerV3
  https://arxiv.org/abs/2301.04104
- Micheli, Alonso, Fleuret, "Transformers are Sample-Efficient World Models" / IRIS
  https://arxiv.org/abs/2209.00588
- Zhang et al., "STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning"
  https://arxiv.org/abs/2310.09615
- Cobbe et al., "Leveraging Procedural Generation to Benchmark Reinforcement Learning" / Procgen
  https://arxiv.org/abs/1912.01588

## Practical MVP architecture

Use this if time is tight:

- `dataset.py`: loads `.npz`, normalizes frames to `[0, 1]`, samples chunks of 32-64 frames.
- `model.py`:
  - CNN encoder: 64x64x3 -> latent vector
  - action embedding: 15 actions -> 32 dims
  - GRU: `[latent, action] -> next hidden`
  - decoder: hidden -> next 64x64 RGB
- Losses:
  - next-frame MSE or BCE
  - optional latent prediction loss
  - optional reward/done prediction heads if recorded
- `train_world_model.py`: teacher-forced next-frame training.
- `play_world_model.py`: load checkpoint, seed with first real frame or reset frame, then keyboard
  actions drive the model at target FPS.

Expected result: a usable but blurry/unstable simulator after long rollouts. For the demo, keep a
short real-frame warmup and periodically reset hidden state if needed.

## Stronger version

If the baseline reaches 10 FPS but looks too blurry:

- Train a VQ-VAE tokenizer over frames.
- Represent each frame as an 8x8 grid of discrete tokens.
- Train a small causal Transformer or GRU to predict next-frame tokens conditioned on action.
- Decode predicted tokens into RGB frames.

This is closer to IRIS and should preserve sharper sprites and level geometry than pixel MSE.

## Evaluation checklist

- Runs interactively without Procgen environment stepping.
- Accepts keyboard actions in real time.
- Renders model-predicted frames, not replayed dataset frames.
- Resolution is 64x64 internally.
- Sustains at least 10 FPS on lab 507 machines.
- Demo records a short video/GIF and logs measured FPS.
- Report includes comparison: real frame, reconstruction, imagined rollout.
