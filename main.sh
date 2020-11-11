#!/usr/bin/env bash
# TODO poly train schedule

#python train.py --model_name DeeperX71_ASPP_CE_Adam_Cleaner_Poly --batch_size 16 \
#    --optim adam --lr 0.00001 --arch deeper.DeeperX71 --loss_type ce \
#    --load_path final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth \
#    --lr_schedule poly --poly_exp 2 --data_filter --retrain

# TODO multi scale DeeperX71 and clean data
#python train.py --model_name DeeperX71_ASPP_Mscale_CE_Adam_Cleaner --batch_size 8 \
#    --optim adam --lr 0.00001 --arch deeper.DeeperX71_ASPP_Mscale --loss_type ce \
#    --load_path final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth \
#    --val_freq 2000 --data_filter --retrain


# TODO rmi loss exp
#python train.py --model_name DeeperX71_ASPP_RMILoss_Adam_Cleaner --batch_size 16 \
#    --optim adam --lr 0.0001 --arch deeper.DeeperX71 --loss_type rmi \
#    --load_path final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth \
#    --val_freq 1000 --data_filter --retrain

# TODO new mean and std
python train.py --model_name DeeperX71_ASPP_CE_Adam_MeanStd --batch_size 16 \
    --optim adam --lr 0.0001 --arch deeper.DeeperX71 --loss_type ce \
    --val_freq 1000


# TODO new transform
python train.py --model_name deeperX71_ASPP_CE_Adam_Poly --batch_size 16 \
    --optim adam --lr 0.00005 --arch deeper.DeeperX71 --loss_type ce \
    --load_path final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth \
    --lr_schedule poly --poly_exp 2 --retrain
