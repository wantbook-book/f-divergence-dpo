from pathlib import Path
import matplotlib.pyplot as plt
def read_f_divergence_and_reward(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    f_divergence = None
    reward = None

    for line in lines:
        if "Averaged f-divergence:" in line:
            f_divergence = float(line.split(":")[1].strip())
        elif "Averaged reward:" in line:
            reward = float(line.split(":")[1].strip())

    return f_divergence, reward

DIVERGENCE_THRES = 70

output_dir = Path(__file__).parent.parent / 'outputs'

# dpo
dpo_divergences = []
dpo_rewards = []
kl_reward_file = output_dir/'dpo_kl_vs_reward.txt'
if kl_reward_file.exists():
    with open(kl_reward_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        f_divergence, reward = list(map(float, line.split()))
        if f_divergence > DIVERGENCE_THRES:
            continue
        dpo_divergences.append(f_divergence)
        dpo_rewards.append(reward)
else:

    dpo_output_dir = Path(__file__).parent.parent / 'outputs' / 'dpo0.1_imdb'
    for file_path in dpo_output_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.txt':
            f_divergence, reward = read_f_divergence_and_reward(file_path)
            # if f_divergence > DIVERGENCE_THRES:
            #     continue
            print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
            dpo_divergences.append(f_divergence)
            dpo_rewards.append(reward)

    with open(output_dir/'dpo_kl_vs_reward.txt', 'w') as file:
        for f_divergence, reward in zip(dpo_divergences, dpo_rewards):
            file.write(f"{f_divergence} {reward}\n")


# sft
sft_divergences = []
sft_rewards = []

kl_reward_file = output_dir/'sft_kl_vs_reward.txt'
if kl_reward_file.exists():
    with open(kl_reward_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        f_divergence, reward = list(map(float, line.split()))
        if f_divergence > DIVERGENCE_THRES:
            continue
        sft_divergences.append(f_divergence)
        sft_rewards.append(reward)
else:
    sft_output_dir = Path(__file__).parent.parent / 'outputs' / 'sft_imdb'
    for file_path in sft_output_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.txt':
            f_divergence, reward = read_f_divergence_and_reward(file_path)
            # if f_divergence > DIVERGENCE_THRES:
            #     continue
            print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
            sft_divergences.append(f_divergence)
            sft_rewards.append(reward)

    with open(output_dir/'sft_kl_vs_reward.txt', 'w') as file:
        for f_divergence, reward in zip(sft_divergences, sft_rewards):
            file.write(f"{f_divergence} {reward}\n")

# gt ppo
gt_ppo_divergences = []
gt_ppo_rewards = []

kl_reward_file = output_dir/'gt_ppo_kl_vs_reward.txt'
if kl_reward_file.exists():
    with open(kl_reward_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        f_divergence, reward = list(map(float, line.split()))
        if f_divergence > DIVERGENCE_THRES:
            continue
        gt_ppo_divergences.append(f_divergence)
        gt_ppo_rewards.append(reward)
else:
    gt_ppo_output_dirs = [Path(__file__).parent.parent / 'outputs' / f'ppo_gt_target{i}_imdb' for i in range(3, 15, 3)]

    for gt_ppo_output_dir in gt_ppo_output_dirs:
        for file_path in gt_ppo_output_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.txt':
                f_divergence, reward = read_f_divergence_and_reward(file_path)
                # if f_divergence > DIVERGENCE_THRES:
                #     continue
                print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
                gt_ppo_divergences.append(f_divergence)
                gt_ppo_rewards.append(reward)

    with open(output_dir/'gt_ppo_kl_vs_reward.txt', 'w') as file:
        for f_divergence, reward in zip(gt_ppo_divergences, gt_ppo_rewards):
            file.write(f"{f_divergence} {reward}\n")

# rm ppo
rm_ppo_divergences = []
rm_ppo_rewards = []

kl_reward_file = output_dir/'rm_ppo_kl_vs_reward.txt'
if kl_reward_file.exists():
    with open(kl_reward_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        f_divergence, reward = list(map(float, line.split()))
        if f_divergence > DIVERGENCE_THRES:
            continue
        rm_ppo_divergences.append(f_divergence)
        rm_ppo_rewards.append(reward)
else:
    rm_ppo_output_dirs = [Path(__file__).parent.parent / 'outputs' / f'ppo_rm_target{i}_imdb' for i in range(3, 6, 3)]

    for rm_ppo_output_dir in rm_ppo_output_dirs:
        for file_path in rm_ppo_output_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.txt':
                f_divergence, reward = read_f_divergence_and_reward(file_path)
                # if f_divergence > DIVERGENCE_THRES:
                #     continue
                print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
                rm_ppo_divergences.append(f_divergence)
                rm_ppo_rewards.append(reward)

    with open(output_dir/'rm_ppo_kl_vs_reward.txt', 'w') as file:
        for f_divergence, reward in zip(rm_ppo_divergences, rm_ppo_rewards):
            file.write(f"{f_divergence} {reward}\n")


# DPO (Ours): #DAA520 (Goldenrod or Orange-yellow)
# Unlikelihood: #468B74 (Teal green)
# PPO (Our impl.): #FF69B4 (Hot pink or Magenta)
# PPO-GT (Our impl.): #FF8C00 (Dark orange)
# PPO-GT (TRL): #9370DB (Medium purple)
# Preferred-FT: #8FBC8F (Dark sea green)
# ax.plot([1, 2, 3], [1, 4, 9], label='DPO (Ours)', color='#DAA520')
# ax.plot([1, 2, 3], [1, 2, 3], label='Unlikelihood', color='#468B74')
# ax.plot([1, 2, 3], [3, 2, 1], label='PPO (Our impl.)', color='#FF69B4')
# ax.plot([1, 2, 3], [2, 3, 1], label='PPO-GT (Our impl.)', color='#FF8C00')
# ax.plot([1, 2, 3], [3, 1, 4], label='PPO-GT (TRL)', color='#9370DB')
# ax.plot([1, 2, 3], [2, 1, 4], label='Preferred-FT', color='#8FBC8F')

plt.figure(figsize=(10,5))
plt.scatter(dpo_divergences, dpo_rewards, color='#DAA520', marker='o', label='DPO')
plt.scatter(sft_divergences, sft_rewards, color='#8FBC8F', marker='o', label='SFT')
plt.scatter(gt_ppo_divergences, gt_ppo_rewards, color='#9370DB', marker='o', label='PPO-GT')
plt.scatter(rm_ppo_divergences, rm_ppo_rewards, color='#FF69B4', marker='o', label='PPO-RM')

# 添加标题和标签
plt.title('Reward vs. f-divergence')
plt.xlabel('f-divergence')
plt.ylabel('Reward')

plt.legend()

# 显示图形
plt.savefig(output_dir/'kl_vs_reward.png')