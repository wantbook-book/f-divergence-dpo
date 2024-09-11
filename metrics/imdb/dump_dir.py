from pathlib import Path

result_dir = Path('/pubshare/fwk/ppo_ckpts/2024-08-22-13-25-target3')
sub_dir = []
for filepath in result_dir.iterdir():
    if filepath.is_dir():
        sub_dir.append(filepath.name)
print('[', end='')
for item in sub_dir:
    print(f'"{item}" ', end='')

print(']')
