cuda=2


env_name="procgen-500"
# env_name="procgen-1000"


finetune_alg="scpiql"

way_steps=3    # original is 3

pretrain_expectile=0.7
explore_temperature=1.0
discount=0.99
data_source="mixed"
frame_skip=1
hiql_finetune_actor_supervision='value'
mixed_finetune_value_loss='hiql'
cql_alpha=0.005

warmup_steps=500
rep_type="concat"
rep_dim=10

high_p_randomgoal=1
num_workers=8

more_args=""
wandb_tags=""

policy_train_rep=1      # whether to tune representation with policy gradients
tune_repr_high_actor=1      # whether to tune representation in value & high actor
tune_high_actor=1   # whether to fix representation but tune high actor
if [ ${policy_train_rep} = 1 ]; then
    more_args="--policy_train_rep 1 --grad_value_repr --high_actor_update ${more_args}"
    wandb_tags="p_grad ${wandb_tags}"
elif [ ${tune_repr_high_actor} = 1 ]; then
    more_args="--grad_value_repr --high_actor_update ${more_args}"
    wandb_tags="grad_r_ha ${wandb_tags}"
elif [ ${tune_high_actor} = 1 ]; then
    more_args="--high_actor_update ${more_args}"
    wandb_tags="ha ${wandb_tags}"
fi

if [ ${way_steps} != 3 ]; then
    wandb_tags="${wandb_tags} way${way_steps}"
fi


job_type="hiql_autockpt"


for seed in `seq 5 5`
do
    CUDA_VISIBLE_DEVICES=${cuda} python train_finetune.py --seed ${seed} --env_name ${env_name} --frame_skip ${frame_skip} \
    --data_source ${data_source} --finetune_alg ${finetune_alg} --hiql_finetune_actor_supervision ${hiql_finetune_actor_supervision} \
    --mixed_finetune_value_loss ${mixed_finetune_value_loss} --cql_alpha ${cql_alpha} \
    --num_workers ${num_workers} --max_env_steps 3000000 --warmup_steps ${warmup_steps} --epoch_steps 1 --replay_buffer_size 300000 --train_iters 2 --explore_temperature ${explore_temperature} \
    --auto_find_checkpoint \
    --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --geom_sample 1 \
    --use_waypoints 1 --way_steps ${way_steps} --high_p_randomgoal ${high_p_randomgoal} \
    --discount ${discount} --temperature 1 --high_temperature 1 --pretrain_expectile ${pretrain_expectile} \
    --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --use_rep 1 --rep_type ${rep_type} --rep_dim ${rep_dim} \
    --visual 1 \
    ${more_args} \
    --batch_size 256 --eval_interval 25000 --save_interval 0 --log_interval 1000 \
    --job_type --wandb_tags ${more_tags} ${wandb_tags} ${data_source} ${mixed_finetune_value_loss} \
    --run_name ${finetune_alg}_${pretrain_expectile}_${mixed_finetune_value_loss}_seed${seed}
done
