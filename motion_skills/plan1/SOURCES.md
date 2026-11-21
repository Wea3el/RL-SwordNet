# Motion sources and usage notice

This project uses two motion references distributed in [NVIDIA ASE](https://github.com/nv-tlabs/ASE), originally provided by **Reallusion**. They are not original motion captures by the RL-SwordNet team, and this package is not endorsed by NVIDIA or Reallusion.

The pinned upstream [motion README](https://github.com/nv-tlabs/ASE/blob/6f9b4f1f289603eee6a4f45d082bc6e6d83fecec/ase/data/motions/reallusion_sword_shield/README.txt) limits the motion data to noncommercial use. Do not infer unrestricted data rights from ASE's code license or public availability. This repository grants no additional rights and does not assert permission to redistribute original or derived motion data. Review the original terms for your intended use; obtain permission where necessary.

The original motion packs linked upstream are [Sword and Shield Stunts](https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-stunts) and [Sword and Shield Moves](https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-moves). Our small fixed-base template adaptations are not validated HEMA or historical Germanic technique demonstrations, nor a dataset of reactive fencing decisions.

## Pinned inputs

Upstream commit: `6f9b4f1f289603eee6a4f45d082bc6e6d83fecec`.

Directory: `ase/data/motions/reallusion_sword_shield`.

- `RL_Avatar_Atk_Stab_Motion.npy`: 178,957 bytes, 128 frames. SHA-256: `84447bd38d1efa356dc44d68f5e5d516760c03cdb236aa1a13c181f3bbc6d3c0`.
- `RL_Avatar_Sword_ParryUp_Motion.npy`: 82,077 bytes, 58 frames. SHA-256: `d883d04247caa594b5a1f142ea18aac930ef7a97603abe22758be386af1d4b99`.

The exact raw URLs, frame rates, conventions, and three action definitions are recorded in [manifest.json](manifest.json). Guard is derived from the first Stab pose; it is not a third upstream file. No motion binary is included in this Git branch. Downloads and derived arrays are generated locally under ignored `outputs/`.

The downloader, restricted numeric interpreter, transformation code, and sampler are implementation tools for this project. The interpreter recognizes a narrow audited serialization layout without importing or executing upstream ASE code. No upstream training implementation or pre-trained policy is included or required.
