python train.py --model_name DeeperX71_ASPP_CE_Adam_Cleaner_Poly --batch_size 16 \
    --optim adam --lr 0.00001 --arch deeper.DeeperX71 --loss_type ce \
    --load_path final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth \
    --lr_schedule poly --poly_exp 2 --data_filter --retrain