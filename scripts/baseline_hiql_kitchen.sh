cuda=2


# env_name="kitchen-partial-v0"
env_name="kitchen-mixed-v0"


finetune_alg="scpiql"

warmup_steps=280

pretrain_expectile=0.7
explore_temperature=1.0
discount=0.99
data_source="mixed"
frame_skip=1
hiql_finetune_actor_supervision='value'
mixed_finetune_value_loss='hiql'
cql_alpha=0.005
way_steps=25

use_rep=0
rep_type="concat"
rep_dim=10


job_type="hiql_autockpt"


for seed in `seq 1 5`
do
    CUDA_VISIBLE_DEVICES=${cuda} python train_finetune.py --seed ${seed} --env_name ${env_name} --frame_skip ${frame_skip} \
    --data_source ${data_source} --finetune_alg ${finetune_alg} --hiql_finetune_actor_supervision ${hiql_finetune_actor_supervision} \
    --mixed_finetune_value_loss ${mixed_finetune_value_loss} --cql_alpha ${cql_alpha} \
    --num_workers 8 --max_env_steps 3000000 --warmup_steps ${warmup_steps} --epoch_steps 1 --replay_buffer_size 300000 --train_iters 2 --explore_temperature ${explore_temperature} \
    --auto_find_checkpoint \
    --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --geom_sample 1 \
    --use_waypoints 1 --way_steps ${way_steps} --high_p_randomgoal 0.3 \
    --discount ${discount} --temperature 1 --high_temperature 1 --pretrain_expectile ${pretrain_expectile} \
    --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --use_rep ${use_rep} --rep_type ${rep_type} --rep_dim ${rep_dim} \
    --policy_train_rep 0 \
    --batch_size 1024 --eval_interval 20000 --save_interval 0 --log_interval 1000 \
    --job_type ${job_type} --wandb_tags ${data_source} ${mixed_finetune_value_loss} \
    --run_name ${finetune_alg}_${pretrain_expectile}_${mixed_finetune_value_loss}_seed${seed}
done
