# stage-one

python main.py \
    --project trans_Full \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --name baseline \
    --cuda_no 0 \
    --do_full_train


#stage-two

python3 main.py --project trans_knn \
    --project "" \
    --task "pdtb2_4" \
    --name "" \
    --cuda_no 2 \
    --epochs 20 \
    --learning_rate 1e-5 \
    --batch_size 32 \
    --logging_steps 100 \
    --seed 0 \
    --do_trans_train