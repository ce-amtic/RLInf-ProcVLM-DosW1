# Archive manifest

## Source revisions

- RLInf baseline: `db677be3a7c45347fd32611adc73ad611c7c18b0`
- ProcVLM public submodule: `377523a31f05bab9c0db5ac8b9edfa7b7f03968a`
- RoboTwin patch base: `0008ae6800df9f75fc8de7098bacb01735fd8fd2`

The repository consists of the RLInf baseline, the complete tracked delta from
the validated 127 working tree, the ProcVLM reward adapter files, and the small
RoboTwin runtime patch required by that deployment.

## Included

- RLInf source and official DOS-W1 support
- ProcVLM reward input, subprocess backend/server, worker, and configuration
- OpenVLA-OFT and pi0.5 ProcVLM-aware launchers
- Validated RoboTwin `move_can_pot` launcher
- Public ProcVLM Git submodule
- RoboTwin source patch, without RoboTwin model/data assets

## Excluded

- ProcVLM, policy, and RL checkpoints
- RoboTwin demonstrations and assets
- Python/Conda/uv virtual environments
- Hugging Face caches, W&B state, logs, outputs, videos, and temporary files
- Machine credentials, tokens, and private working-tree metadata

