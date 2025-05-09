for i in {0..3}; do
    python main.py \
        --gpu_id $i \
        --data_pths /cmlscratch/sjaelee/Workspace/RPL/dataset/ThinkLite-VL-Hard-11k.parquet \
        --additional_eval_models Qwen/Qwen3-4B allenai/OLMo-2-0425-1B \
        --chunk_idx $i \
        --num_chunks 4 \
        --do_train \
        --output_file output_$i.parquet &
done