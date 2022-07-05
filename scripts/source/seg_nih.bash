python main_source.py seg_nih \
    -G $1 \
    --method seg_train \
    --train_list NIH_train \
    --val_list NIH_val \
    --data_root /export/ccvl12b/yyao/medical_data/NIH \
    --val_data_root /export/ccvl12b/yyao/medical_data/NIH \
    --data_path /export/ccvl12b/yyao/medical_data/Multi_all.json \
    --eval_epoch 20 \
    --save_epoch 800 \
    --max_epoch 2400 \