TRANSFORMERS_OFFLINE=1 python3 main.py \
    --mode train \
    --relation_dim 200 \
    --do_batch_norm 1 \
    --gpu 2 \
    --freeze 1 \
    --batch_size 16 \
    --validate_every 10 \
    --hops webqsp_half \       
    --lr 0.00002 \
    --entdrop 0.0 \
    --reldrop 0.0 \
    --scoredrop 0.0 \
    --decay 1.0 \
    --model ComplEx \
    --patience 20 \
    --ls 0.05 \
    --l3_reg 0.001 \
    --nb_epochs 200 \
    --outfile half_fbwq