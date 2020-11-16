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
#python train.py --model_name DeeperX71_ASPP_CE_Adam_MeanStd --batch_size 16 \
#    --optim adam --lr 0.0001 --arch deeper.DeeperX71 --loss_type ce \
#    --val_freq 1000


## TODO new transform with poly and ce
#python train.py --model_name deeperX71_ASPP_CE_Adam_Cleaner_Poly2 --batch_size 16 \
#    --optim adam --lr 0.00005 --arch deeper.DeeperX71 --loss_type ce \
#    --load_path final_logs/DeeperX71_ASPP_CE_Adam_Cleaner_Poly/model-step-340999.pth \
#    --lr_schedule poly --poly_exp 2 --retrain --data_filter


##
#python train.py --model_name deeperX71_ASPP_Pool_CE_Adam --batch_size 16 \
#    --optim adam --lr 0.0001 --arch deeper.DeeperX71_ASPP_Pool --loss_type ce

##
#python train.py --model_name deeperX71_ASPP_CE_SGD_Cosine --batch_size 16 \
#    --optim sgd --lr 0.001 --arch deeper.DeeperX71 --loss_type ce \
#    --load_path final_logs/DeeperX71_ASPP_CE_Adam_Cleaner_Poly/model-step-340999.pth \
#    --lr_schedule CosineAnnealingLR --val_freq 5000 --retrain

##
python train.py --model_name deeperX71_ASPP_CE_SGD_Cosine --batch_size 16 \
    --optim sgd --lr 0.001 --arch deeper.DeeperX71 --loss_type ce \
    --load_path final_logs/DeeperX71_ASPP_CE_Adam_Cleaner_Poly/model-step-340999.pth \
    --lr_schedule CosineAnnealingLR --val_freq 5000 --retrain --data_filter



