#!/bin/bash


MAX_ROUND=1
# train_dataset=U,P
conv=LDC_M
EPOCH_train=25
EPOCH_finetune=50
EPOCH_cGAN=500
model_S=2


for round in $(seq 1 $MAX_ROUND)
do
    echo Round \#"$round"


    for train_dataset in C
    do
        echo python3 train_label.py --train_dataset=$train_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs 15 #--do_not_preload
        python3 train_label.py --train_dataset $train_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs 15 #--do_not_preload

        echo train_cGAN.py --train_dataset $train_dataset --conv $conv --model_S $model_S --epoch $EPOCH_cGAN --bs 100
        python3 train_cGAN.py --train_dataset $train_dataset --conv $conv --model_S $model_S --epoch $EPOCH_cGAN --bs 100

        for test_dataset in P
        do

            bs=1
        
            # echo python3 test_tent_cGAN.py --train_dataset $train_dataset --test_dataset $test_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs $bs --do_not_preload --testFold 0 --do_not_adapt
            # python3 test_tent_cGAN.py --train_dataset $train_dataset --test_dataset $test_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs $bs --do_not_preload --testFold 0 --do_not_adapt
        
            echo python3 test_tent_cGAN.py --train_dataset $train_dataset --test_dataset $test_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs $bs --testFold 1 #--do_not_preload 
            python3 test_tent_cGAN.py --train_dataset $train_dataset --test_dataset $test_dataset --conv $conv --model_S $model_S --epoch $EPOCH_train --bs $bs --testFold 1 #--do_not_preload 
        
        
        done
    done
    
done
