# scale 4
python fno_fluid.py \
    --data_path "../CAKRes/DatasetRe16k/train/nskt_Re16000-002.h5" --val_data_path  "../CAKRes/DatasetRe16k/valid/nskt_Re16000-003.h5" \
    --exp_name train_valid_all_s4_t5_n2_dr0.1_b8_lr0.001 \
    --scale 4 \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.001 --sample_limit -1 \
    --t_skip 5 \
    --n_skip 2
    


# scale 8
python fno_fluid.py \
    --data_path "../CAKRes/DatasetRe16k/train/nskt_Re16000-002.h5" --val_data_path  "../CAKRes/DatasetRe16k/valid/nskt_Re16000-003.h5" \
    --exp_name train_valid_all_s8_t5_n2_dr0.1_b8_lr0.001 \
    --scale 8 \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.001 --sample_limit -1 \
    --t_skip 5 \
    --n_skip 2

python fno_fluid.py \
    --data_path "../CAKRes/DatasetRe16k/train/nskt_Re16000-002.h5" --val_data_path  "../CAKRes/DatasetRe16k/valid/nskt_Re16000-003.h5" \
    --exp_name train_valid_all_s16_t5_n2_dr0.1_b8_lr0.001 \
    --scale 16 \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.001 --sample_limit -1 \
    --t_skip 5 \
    --n_skip 2
