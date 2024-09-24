
from pathlib import Path
import matplotlib.pyplot as plt
from io_utils import read_all_divergence_and_reward
def draw_divergence_and_reward(paths: Path, labels: list[str], colors: list[str], divergence_name: str, output_dir:Path):
    all_divergences = []
    all_rewards = []
    for path in paths:
        divergences, rewards = read_all_divergence_and_reward(path, output_dir=output_dir)
        all_divergences.append(divergences)
        all_rewards.append(rewards)
    
    plt.figure(figsize=(10,5))
    for divergences, rewards, label, color in zip(all_divergences, all_rewards, labels, colors):
        plt.scatter(divergences, rewards, color=color, marker='o', label=label)

    # 添加标题和标签
    plt.title(f'Reward vs {divergence_name}')
    plt.xlabel(divergence_name)
    plt.ylabel('Reward')

    plt.legend()

    # 显示图形
    plt.savefig(output_dir/f'{output_dir.name}_{divergence_name}_vs_reward.png')
    print('save divergence vs reward figure to '+ str(output_dir/f'{output_dir.name}_{divergence_name}_vs_reward.png'))

    

if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent/'outputs'
    

    labels = ['DPO', 'SFT', 'PPO-GT', 'PPO-RM', 'Unlikelihood']
    colors = ['#DAA520', '#8FBC8F', '#9370DB', '#FF69B4', '#00CC99']

    # 深粉色 #FF0066
    # 橙色 #FFA500
    # 蓝色 #0066CC
    paths = [
        # root_dir / 'graphs/dpo_kl_vs_reward.txt',
        # root_dir / 'graphs/sft_kl_vs_reward.txt',
        # root_dir / 'graphs/gt_ppo_kl_vs_reward.txt',
        # root_dir / 'graphs/rm_ppo_kl_vs_reward.txt',
        root_dir / 'unlike_imdb1.0'
    ]
    # labels = ['DPO', 'SFT']
    # colors = ['#DAA520', '#8FBC8F']
    divergence_name = 'rkl_divergence'

    paths = [
        # root_dir / 'jsd_imdb',
        # root_dir / 'forward_imdb',
        # root_dir / 'graphs/alpha0.1_imdb.txt',
        root_dir / 'unlike_imdb1.0'
    ]
    colors = ['#FF0066']
    colors = ['#00CC99']
    labels = [
        # 'DPO-JSD',
        # 'DPO-FKL',
        # r'DPO($\alpha$=0.1)',
        'Unlikelihood'
    ]

    divergence_name = 'JSD'
    divergence_name = 'Forward KL'
    divergence_name = r'$\alpha$-divergence(0.1)'
    divergence_name = 'RKL'


    output_dir = root_dir/'graphs'
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_divergence_and_reward(
        paths=paths,
        labels=labels,
        colors=colors,
        divergence_name=divergence_name,
        output_dir=output_dir
    )
    




# output_dir = Path(__file__).parent.parent / 'outputs'

# # dpo
# dpo_divergences = []
# dpo_rewards = []
# kl_reward_file = output_dir/'dpo_kl_vs_reward.txt'
# if kl_reward_file.exists():
#     with open(kl_reward_file, 'r') as file:
#         lines = file.readlines()
#     for line in lines:
#         f_divergence, reward = list(map(float, line.split()))
#         if f_divergence > DIVERGENCE_THRES:
#             continue
#         dpo_divergences.append(f_divergence)
#         dpo_rewards.append(reward)
# else:

#     dpo_output_dir = Path(__file__).parent.parent / 'outputs' / 'dpo0.1_imdb'
#     for file_path in dpo_output_dir.iterdir():
#         if file_path.is_file() and file_path.suffix == '.txt':
#             f_divergence, reward = read_f_divergence_and_reward(file_path)
#             # if f_divergence > DIVERGENCE_THRES:
#             #     continue
#             print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
#             dpo_divergences.append(f_divergence)
#             dpo_rewards.append(reward)

#     with open(output_dir/'dpo_kl_vs_reward.txt', 'w') as file:
#         for f_divergence, reward in zip(dpo_divergences, dpo_rewards):
#             file.write(f"{f_divergence} {reward}\n")


# # sft
# sft_divergences = []
# sft_rewards = []

# kl_reward_file = output_dir/'sft_kl_vs_reward.txt'
# if kl_reward_file.exists():
#     with open(kl_reward_file, 'r') as file:
#         lines = file.readlines()
#     for line in lines:
#         f_divergence, reward = list(map(float, line.split()))
#         if f_divergence > DIVERGENCE_THRES:
#             continue
#         sft_divergences.append(f_divergence)
#         sft_rewards.append(reward)
# else:
#     sft_output_dir = Path(__file__).parent.parent / 'outputs' / 'sft_imdb'
#     for file_path in sft_output_dir.iterdir():
#         if file_path.is_file() and file_path.suffix == '.txt':
#             f_divergence, reward = read_f_divergence_and_reward(file_path)
#             # if f_divergence > DIVERGENCE_THRES:
#             #     continue
#             print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
#             sft_divergences.append(f_divergence)
#             sft_rewards.append(reward)

#     with open(output_dir/'sft_kl_vs_reward.txt', 'w') as file:
#         for f_divergence, reward in zip(sft_divergences, sft_rewards):
#             file.write(f"{f_divergence} {reward}\n")

# # gt ppo
# gt_ppo_divergences = []
# gt_ppo_rewards = []

# kl_reward_file = output_dir/'gt_ppo_kl_vs_reward.txt'
# if kl_reward_file.exists():
#     with open(kl_reward_file, 'r') as file:
#         lines = file.readlines()
#     for line in lines:
#         f_divergence, reward = list(map(float, line.split()))
#         if f_divergence > DIVERGENCE_THRES:
#             continue
#         gt_ppo_divergences.append(f_divergence)
#         gt_ppo_rewards.append(reward)
# else:
#     gt_ppo_output_dirs = [Path(__file__).parent.parent / 'outputs' / f'ppo_gt_target{i}_imdb' for i in range(3, 15, 3)]

#     for gt_ppo_output_dir in gt_ppo_output_dirs:
#         for file_path in gt_ppo_output_dir.iterdir():
#             if file_path.is_file() and file_path.suffix == '.txt':
#                 f_divergence, reward = read_f_divergence_and_reward(file_path)
#                 # if f_divergence > DIVERGENCE_THRES:
#                 #     continue
#                 print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
#                 gt_ppo_divergences.append(f_divergence)
#                 gt_ppo_rewards.append(reward)

#     with open(output_dir/'gt_ppo_kl_vs_reward.txt', 'w') as file:
#         for f_divergence, reward in zip(gt_ppo_divergences, gt_ppo_rewards):
#             file.write(f"{f_divergence} {reward}\n")

# # rm ppo
# rm_ppo_divergences = []
# rm_ppo_rewards = []

# kl_reward_file = output_dir/'rm_ppo_kl_vs_reward.txt'
# if kl_reward_file.exists():
#     with open(kl_reward_file, 'r') as file:
#         lines = file.readlines()
#     for line in lines:
#         f_divergence, reward = list(map(float, line.split()))
#         if f_divergence > DIVERGENCE_THRES:
#             continue
#         rm_ppo_divergences.append(f_divergence)
#         rm_ppo_rewards.append(reward)
# else:
#     rm_ppo_output_dirs = [Path(__file__).parent.parent / 'outputs' / f'ppo_rm_target{i}_imdb' for i in range(3, 6, 3)]

#     for rm_ppo_output_dir in rm_ppo_output_dirs:
#         for file_path in rm_ppo_output_dir.iterdir():
#             if file_path.is_file() and file_path.suffix == '.txt':
#                 f_divergence, reward = read_f_divergence_and_reward(file_path)
#                 # if f_divergence > DIVERGENCE_THRES:
#                 #     continue
#                 print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
#                 rm_ppo_divergences.append(f_divergence)
#                 rm_ppo_rewards.append(reward)

#     with open(output_dir/'rm_ppo_kl_vs_reward.txt', 'w') as file:
#         for f_divergence, reward in zip(rm_ppo_divergences, rm_ppo_rewards):
#             file.write(f"{f_divergence} {reward}\n")


# # DPO (Ours): #DAA520 (Goldenrod or Orange-yellow)
# # Unlikelihood: #468B74 (Teal green)
# # PPO (Our impl.): #FF69B4 (Hot pink or Magenta)
# # PPO-GT (Our impl.): #FF8C00 (Dark orange)
# # PPO-GT (TRL): #9370DB (Medium purple)
# # Preferred-FT: #8FBC8F (Dark sea green)
# # ax.plot([1, 2, 3], [1, 4, 9], label='DPO (Ours)', color='#DAA520')
# # ax.plot([1, 2, 3], [1, 2, 3], label='Unlikelihood', color='#468B74')
# # ax.plot([1, 2, 3], [3, 2, 1], label='PPO (Our impl.)', color='#FF69B4')
# # ax.plot([1, 2, 3], [2, 3, 1], label='PPO-GT (Our impl.)', color='#FF8C00')
# # ax.plot([1, 2, 3], [3, 1, 4], label='PPO-GT (TRL)', color='#9370DB')
# # ax.plot([1, 2, 3], [2, 1, 4], label='Preferred-FT', color='#8FBC8F')

# plt.figure(figsize=(10,5))
# plt.scatter(dpo_divergences, dpo_rewards, color='#DAA520', marker='o', label='DPO')
# plt.scatter(sft_divergences, sft_rewards, color='#8FBC8F', marker='o', label='SFT')
# plt.scatter(gt_ppo_divergences, gt_ppo_rewards, color='#9370DB', marker='o', label='PPO-GT')
# plt.scatter(rm_ppo_divergences, rm_ppo_rewards, color='#FF69B4', marker='o', label='PPO-RM')

# # 添加标题和标签
# plt.title('Reward vs. f-divergence')
# plt.xlabel('f-divergence')
# plt.ylabel('Reward')

# plt.legend()

# # 显示图形
# plt.savefig(output_dir/'kl_vs_reward.png')