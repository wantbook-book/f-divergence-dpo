import os
import re

# 设置你的文件目录路径和输出文件路径
input_dir = '/pubshare/fwk/code/f-divergence-dpo/outputs/0530_gamma_dpo0.1_imdb'  # 替换为你的目录路径
output_file = f'{input_dir}/results.txt'

# 正则表达式
reward_pattern = re.compile(r'reward:\s*([0-9.]+)')
fdiv_pattern = re.compile(r'f-divergence:\s*([0-9.]+)')
filename_pattern = re.compile(r'.*_step-\d+$')

# 存储结果
results = []

for filename in os.listdir(input_dir):
    # 如果filename是
    if filename_pattern.match(filename):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                reward_match = reward_pattern.search(content)
                fdiv_match = fdiv_pattern.search(content)
                if reward_match and fdiv_match:
                    reward = reward_match.group(1)
                    fdiv = fdiv_match.group(1)
                    results.append(f"{fdiv} {reward}")

# 写入结果文件
with open(output_file, 'w') as f:
    for line in results:
        f.write(line + '\n')
