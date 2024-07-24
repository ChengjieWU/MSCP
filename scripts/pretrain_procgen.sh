cuda=0


env_name="procgen-500"
# env_name="procgen-1000"


pretrain_alg="scpiql"

pretrain_expectile=0.7
way_steps=3
rep_type="concat"
rep_dim=10
policy_train_rep=0

high_p_randomgoal=0


for seed in `seq 1 5
do
    CUDA_VISIBLE_DEVICES=${cuda} python main.py --seed ${seed} --env_name ${env_name} \
    --pretrain_alg ${pretrain_alg} --pretrain_steps 500002 \
    --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --geom_sample 1 \
    --use_waypoints 1 --way_steps ${way_steps} --high_p_randomgoal ${high_p_randomgoal} \
    --discount 0.99 --temperature 1 --high_temperature 1 --pretrain_expectile ${pretrain_expectile} \
    --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --use_rep 1 --rep_type ${rep_type} --rep_dim ${rep_dim} --policy_train_rep ${policy_train_rep} \
    --visual 1 \
    --batch_size 256 --eval_interval 50000 --save_interval 0 --log_interval 500 \
    --job_type pretrain \
    --run_name seed${seed}
done
