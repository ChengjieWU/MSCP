cuda=0


env_name="calvin"


pretrain_alg="scpiql"
pretrain_expectile=0.7
way_steps=25
use_rep=1
rep_type="concat"
rep_dim=10


for seed in `seq 1 5`
do
    CUDA_VISIBLE_DEVICES=${cuda} python main.py --seed ${seed} --env_name ${env_name} \
    --pretrain_alg ${pretrain_alg} --pretrain_steps 500002 \
    --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --geom_sample 1 \
    --use_waypoints 1 --way_steps ${way_steps} --high_p_randomgoal 0.3 \
    --discount 0.99 --temperature 1 --high_temperature 1 --pretrain_expectile ${pretrain_expectile} \
    --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --use_rep ${use_rep} --rep_type ${rep_type} --rep_dim ${rep_dim} --policy_train_rep 0 \
    --batch_size 1024 --eval_interval 100000 --save_interval 0 --log_interval 500 \
    --job_type pretrain \
    --run_name seed${seed}
done
