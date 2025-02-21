# ThinkPO: Thinking Preference Optimization
> a simple yet effective postSFT method that enhances long CoT reasoning without requiring new long CoT responses

<img src="./imgs/Training_pipeline.png" alt="Training Pipeline">

---
# News
- 2025-02-21: We released our four models: [DeepSeek-R1-Distill-Qwen-7B-ThinkPO](https://huggingface.co/VanWang/DeepSeek-R1-Distill-Qwen-7B-ThinkPO), [Bespoke-Stratos-7B-ThinkPO](https://huggingface.co/VanWang/Bespoke-Stratos-7B-ThinkPO),
[Bespoke-Stratos-7B-repro-SFT](https://huggingface.co/VanWang/Bespoke-Stratos-7B-repro-SFT), [Bespoke-Stratos-7B-repro-ThinkPO](https://huggingface.co/VanWang/Bespoke-Stratos-7B-repro-ThinkPO)

- 2025-02-19: We released our [paper](https://arxiv.org/abs/2502.13173).

---
# Quick Use

## SFT Train
- you could run the following command
```shell
deepspeed train/sft_train.py --model_name Qwen2.5-7B --gradient_accumulation_steps 16 --dataset_name NuminaMath --epoch 3 --lr 1e-5
```

## ThinkPO Train
- you could run the following command
```shell
deepspeed train/dpo_train.py --lr 3e-7 --beta 0.01 --model OpenThinker-7B --dataset Bespoke_dpo_long --gradient_accumulation_steps 12
```

## eval the model
- you could run the following command
```shell
python ./tools/eval.py --model your_model_path --evals MATH500,AIME,GPQADiamond,GSM8K,OlympiadBenchMath --tp 2 --output_file save_result_path.txt
```

## citation
```bibtex
@misc{yang2025thinkingpreferenceoptimization,
      title={Thinking Preference Optimization}, 
      author={Wang Yang and Hongye Jin and Jingfeng Yang and Vipin Chaudhary and Xiaotian Han},
      year={2025},
      eprint={2502.13173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.13173}, 
}
```