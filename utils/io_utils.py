from pathlib import Path
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

def read_all_divergence_and_reward_from_dir(dir_path: Path, output_dir: Path)->tuple[list[float], list[float]]:
    divergences, rewards = [], []

    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.txt':
            f_divergence, reward = read_f_divergence_and_reward(file_path)
            # if f_divergence > DIVERGENCE_THRES:
            #     continue
            # print(f"{file_path.name}: f-divergence: {f_divergence}, reward: {reward}")
            divergences.append(f_divergence)
            rewards.append(reward)
    print('read all divergences and rewards from dir: '+str(dir_path))
    with open(output_dir/f'{dir_path.name}.txt', 'w') as file:
        for f_divergence, reward in zip(divergences, rewards):
            file.write(f"{f_divergence} {reward}\n")
    print('save all divergences and rewards to '+str(output_dir/f'{dir_path.name}.txt'))
    return divergences, rewards

DIVERGENCE_THRES = 70
def read_all_divergence_and_reward_from_file(file_path: Path)->tuple[list[float], list[float]]:
    divergences, rewards = [], []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        f_divergence, reward = list(map(float, line.split()))
        if f_divergence > DIVERGENCE_THRES:
            continue
        divergences.append(f_divergence)
        rewards.append(reward)
    print('read all divergences and rewards from file: '+str(file_path))
    return divergences, rewards

def read_all_divergence_and_reward(path: Path, output_dir:Path)->tuple[list[float], list[float]]:
    if path.is_file():
        return read_all_divergence_and_reward_from_file(path)
    else:
        return read_all_divergence_and_reward_from_dir(path, output_dir)

def read_divergence_and_steps_from_dir(dir_path: Path, output_dir: Path)->tuple[list[float], list[int]]:
    steps = []
    divergences = []

    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.txt':
            f_divergence, reward = read_f_divergence_and_reward(file_path)
            steps.append(int(file_path.stem.split('-')[-1]))
            divergences.append(f_divergence)
    print('read all divergences from dir: '+str(dir_path))
    with open(output_dir/f'{dir_path.name}_divergence_and_steps.txt', 'w') as file:
        for step, f_divergence in zip(steps, divergences):
            file.write(f"{step} {f_divergence}\n")
    print('save all divergences to '+str(output_dir/f'{dir_path.name}_divergence_and_steps.txt'))
    return divergences, steps