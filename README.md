# From-Pixels-to-Semantic
From Pixels to Semantics: A Multi-Stage AI Framework for Structural Damage Detection in Satellite Imagery


### CLIPScore Comparison on xBD Disaster Subset (Full Images)

| Disaster Type         | VLM Model                  | Avg. CLIPScore (%) | Max. CLIPScore | Min. CLIPScore |
| --------------------- | -------------------------- | ------------------ | -------------- | -------------- |
| **xBD**               | VLCE (LLaVA-baseline) [1]  | 55.34              | -              | -              |
|                       | VLCE (QwenVL-baseline) [1] | 60.60              | -              | -              |
| **Moore Tornado**     | Qwen3-vl:32b               | **63.34**          | **72.60**      | **54.83**      |
|                       | Qwen3-vl:8b                | 62.87              | 70.42          | 51.40          |
|                       | Gemma3:27b                 | 60.02              | 70.69          | 50.23          |
|                       | Gemma3:12b                 | 60.02              | 68.55          | 51.80          |
| **Matthew Hurricane** | Qwen3-vl:32b               | **62.42**          | **81.04**      | 50.18          |
|                       | Qwen3-vl:8b                | 62.17              | 77.56          | **51.60**      |
|                       | Gemma3:27b                 | 58.18              | 67.72          | 47.19          |
|                       | Gemma3:12b                 | 57.06              | 67.96          | 44.82          |

---


## Citation

If you use this work, please cite:

```bibtex
@article{shakya2026pixels,
  title={From Pixels to Semantics: A Multi-Stage AI Framework for Structural Damage Detection in Satellite Imagery},
  author={Shakya, Bijay and Hoier, Catherine and Ahmed, Khandaker Mamun},
  journal={arXiv preprint arXiv:2603.22768},
  year={2026}
}
