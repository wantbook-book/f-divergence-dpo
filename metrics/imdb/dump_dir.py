from pathlib import Path

result_dir = Path('/pubshare/fwk/dpo_cache/jovyan/imdb_unlike1.0_gpt2_large_2024-09-25_01-50-04_583973')
sub_dir = []
for filepath in result_dir.iterdir():
    if filepath.is_dir():
        sub_dir.append(filepath.name)
print('[', end='')
for item in sub_dir:
    print(f'"{item}" ', end='')

print(']')
