# # 在简单级别训练，在困难级别测试
# export CUDA_VISIBLE_DEVICES=1
# python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py \
#   train_levels=l \
#   eval=eval_auto \
#   env_size=l \
#   'train_levels.train_levels_list=["l/grasp_easy.json"]' \
#   'eval.eval_levels=["l/grasp_hard.json"]'
  
# # 如果想在训练和测试中都使用困难级别（原始配置）
# export CUDA_VISIBLE_DEVICES=0
# python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py \
#   train_levels=l \
#   eval=eval_auto \
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


python3 /proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/experiments/ppo.py train_levels=random \
  eval=eval_auto \
  'eval.eval_levels=["l/grasp_hard.json"]' \
  learning.ent_coef=0.1 \
  learning.clip_eps=0.3 \
  learning.peak_lr=5e-4

