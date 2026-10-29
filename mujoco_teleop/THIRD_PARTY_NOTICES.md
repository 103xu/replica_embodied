# Third-party sources

## Every-Embodied

The unmodified files under `vendor/every_embodied/` are selected from
<https://github.com/datawhalechina/every-embodied> at commit
`14d7c6c93debf01419b21e50b649a06f15881c38`.

Attribution: Datawhale and contributors. The repository-level license is
Creative Commons Attribution 4.0 International; the original license text is
preserved at `vendor/every_embodied/LICENSE`.

The upstream tutorial credits the following sources for its code and assets:

- OMY robot: ROBOTIS, <https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie>.
- MuJoCo parser: <https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3>.
- Mug and plate: Objaverse, <https://objaverse.allenai.org/>.

These source attributions are retained; the upstream repository-level license
does not replace the original terms of independently sourced assets.

## LeRobot

Runtime dependency from <https://github.com/huggingface/lerobot>, fixed at commit
`10b7b3532543b4adfb65760f02a49b4c537afde7`, copyright Hugging Face,
Apache License 2.0. Installed locally into `.deps` by the installation script;
not redistributed as project source.

## Local changes

`src/omy_repro/` adapts the tutorial workflow into command-line programs.
It adds fixed 20 Hz stepping, local dependency loading, Windows-compatible
image writing, safe dataset directory handling, explicit initial-state snapshots,
complete integration-state restoration after IK, consistent float32 command
execution/recording, and separate interface-smoke, keyboard demonstration,
and scripted IK demonstration modes.

Sections 3-4 additionally use the pinned LeRobot ACT policy (Apache-2.0,
Tony Z. Zhao and the Hugging Face team). The training initialization uses
the official torchvision ResNet18 ImageNet weights from
<https://download.pytorch.org/models/resnet18-f37072fd.pth>.
The local ACT implementation wraps that policy; it does not replace the
underlying ACT network with a different controller.
The original vendor sources are retained without local edits.
