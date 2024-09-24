from pathlib import Path
import matplotlib.pyplot as plt
from io_utils import read_all_divergence_and_reward, read_divergence_and_steps_from_dir

def draw_divergence(path:Path, label, color, output_dir:Path):
    

    divergences, steps = read_divergence_and_steps_from_dir(path, output_dir=output_dir)

    plt.figure(figsize=(10,5))
    plt.scatter(steps, divergences, color=color, marker='o', label=label)

    # 添加标题和标签
    plt.title(f'Divergences')
    plt.xlabel('step')
    plt.ylabel('divergence')

    plt.legend()

    # 显示图形
    plt.savefig(output_dir/f'{output_dir.name}_divergence_by_steps.png')
    print('save divergences figure to '+ str(output_dir/f'{output_dir.name}_{label}_divergence_by_steps.png'))
    
if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent/'outputs'
    paths = [
        # root_dir / 'jsd_imdb',
        root_dir / 'forward_imdb',
        # root_dir / 'alpha0.1_imdb',
    ]
    colors = ['#FF0066']
    labels = [
        # 'DPO-JSD',
        'DPO-FKL',
        # r'DPO($\alpha$=0.1)'
    ]
    path = paths[0]
    color = colors[0]
    label = labels[0]

    output_dir = root_dir/'graphs'
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_divergence(
        path=path,
        color=color,
        label=label,
        output_dir=output_dir
    )