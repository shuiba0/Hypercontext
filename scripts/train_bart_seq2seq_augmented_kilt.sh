#/bin/bash
python scripts/train_bart_seq2seq_augmented_kilt.py \
    --gpus 1 \
#    --accelerator ddp \ 把这行去掉就能跑
    --num_workers 32 \
    --batch_size 64 \
    --max_steps 200000 \
    --divergences kl \
    --train_data_path ./knowledge_editor/datasets/structured_zeroshot-train-new_annotated_final.jsonl \
    --use_views \
    2>&1 | tee models/bart_seq2seq_augmented_structured_zeroshot/log.txt
