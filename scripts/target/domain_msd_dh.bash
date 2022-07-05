python main_target.py domain_msd_dh \
    -G $1 \
    --method domain_adaptation \
    --load_prefix seg_nih \
    --load_prefix_vae vae_nih \
    --train_list MSD_train \
    --val_list MSD_val \
    --data_root <Your_data_path> \
    --val_data_root <Your_data_path> \
    --data_path data/Multi_all.json \
    --pan_index 10 \
    --lambda_vae 1.0 \
    --domain_loss_type 8 \
    --eval_epoch 2 \
    --save_epoch 100 \
    --max_epoch 50