# # 在简单级别训练，在困难级别测试
# export CUDA_VISIBLE_DEVICES=1
python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py \
  train_levels=l \
  eval=eval_general \
  env.dense_reward_scale=0.5 \
  env_size=l \
  'train_levels.train_levels_list=["l/grasp_easy.json"]' \
  'eval.eval_levels=["l/grasp_hard.json"]'
  
# # 如果想在训练和测试中都使用困难级别（原始配置）
# export CUDA_VISIBLE_DEVICES=0
# python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py \
#   train_levels=l \
#   eval=eval_general \
#   env_size=l \
#   'train_levels.train_levels_list=["l/grasp_easy.json","l/grasp_hard.json"]' \
#   'eval.eval_levels=["l/grasp_hard.json"]'

# # 使用全部levels训练，在grasp_hard测试
# export CUDA_VISIBLE_DEVICES=1
# python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py \
#   train_levels=train_all \
#   eval=eval_auto \
#   env_size=l \
#   'eval.eval_levels=["l/grasp_hard.json"]'


# python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py train_levels=random \
#   eval=eval_auto \
#   'eval.eval_levels=["l/grasp_hard.json"]' 


# 使用curriculum训练
python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l \
  eval=eval_general \
  'eval.eval_levels=["l/grasp_hard.json"]'


python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l \
  env.dense_reward_scale=0.7 \
  eval=eval_general \
  'eval.eval_levels=["l/grasp_hard.json"]'

python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l \
  eval=eval_general \
  'eval.eval_levels=["l/grasp_hard.json"]'


python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l eval=eval_general 'eval.eval_levels=["l/grasp_hard.json"]' \
  learning.update_epochs=12 learning.num_minibatches=16 \
  learning.warmup_lr=true learning.initial_lr=1e-5 learning.peak_lr=3e-4 \
  learning.ent_coef=0.005 env.dense_reward_scale=0.5 \
  learning.num_steps=32


python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l eval=eval_general 'eval.eval_levels=["l/grasp_hard.json", "l/grasp_levels/level_5/grasp_level5_v01.json","l/grasp_levels/level_5/grasp_level5_v02.json","l/grasp_levels/level_5/grasp_level5_v03.json","l/grasp_levels/level_5/grasp_level5_v04.json","l/grasp_levels/level_5/grasp_level5_v05.json","l/grasp_levels/level_5/grasp_level5_v06.json","l/grasp_levels/level_5/grasp_level5_v07.json","l/grasp_levels/level_5/grasp_level5_v08.json","l/grasp_levels/level_5/grasp_level5_v09.json","l/grasp_levels/level_5/grasp_level5_v10.json"]' \
  learning.update_epochs=12 learning.num_minibatches=16 \
  learning.warmup_lr=true learning.initial_lr=1e-5 learning.peak_lr=3e-4 \
  learning.ent_coef=0.005 env.dense_reward_scale=0.3 \
  learning.num_steps=32

# 使用末端-球距离 shaping 的示例（设置系数>0启用）
python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo_curriculum.py \
  env_size=l eval=eval_general 'eval.eval_levels=["l/grasp_hard.json", "l/grasp_levels/level_5/grasp_level5_v01.json","l/grasp_levels/level_5/grasp_level5_v02.json","l/grasp_levels/level_5/grasp_level5_v03.json","l/grasp_levels/level_5/grasp_level5_v04.json","l/grasp_levels/level_5/grasp_level5_v05.json","l/grasp_levels/level_5/grasp_level5_v06.json","l/grasp_levels/level_5/grasp_level5_v07.json","l/grasp_levels/level_5/grasp_level5_v08.json","l/grasp_levels/level_5/grasp_level5_v09.json","l/grasp_levels/level_5/grasp_level5_v10.json"]' \
  env.dense_reward_scale=0.5 env.effector_ball_dense_reward_scale=0.5 \
  learning.num_steps=32 learning.update_epochs=12 learning.num_minibatches=16