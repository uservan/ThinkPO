# ThinkPO

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