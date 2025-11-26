import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dump(dataset: str, model: str):
    script_dir = Path(__file__).resolve().parent
    dump_dir = script_dir / 'output_dump' / model
    dump_file = dump_dir / f'{dataset}_chatgpt_api_dump.json'
    if not dump_file.exists():
        raise FileNotFoundError(f"Dump not found at {dump_file}. Run inference and merge fragments first.")
    with dump_file.open('r') as f:
        return json.load(f), dump_dir


def main():
    parser = argparse.ArgumentParser(description="Plot trajectories from a merged zero-shot dump.")
    parser.add_argument('--dataset', default=0, type=int, help="dataset id: 0 eth, 1 hotel, 2 univ, 3 zara1, 4 zara2")
    parser.add_argument('--model', default=0, type=int, help="model id: 0 gpt-3.5-turbo-0301, 1 gpt-4-0314, 2 gpt-3.5-turbo-1106, 3 gpt-4-1106-preview, 4 azure/gpt-4.1")
    parser.add_argument('--scene_idx', default=0, type=int, help="scene index to plot")
    parser.add_argument('--ped_idx', default=0, type=int, help="pedestrian index to plot")
    parser.add_argument('--output', default=None, type=str, help="output png path (defaults to zero-shot/output_dump/<dataset>_sceneXXXX_pedY.png)")
    args = parser.parse_args()

    dataset = ['eth', 'hotel', 'univ', 'zara1', 'zara2'][args.dataset]
    model = ['gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'azure/gpt-4.1'][args.model]

    dump, dump_dir = load_dump(dataset, model)
    scenes = sorted(dump['data'].keys(), key=lambda x: int(x.split('_')[-1]))
    if args.scene_idx >= len(scenes):
        raise IndexError(f"scene_idx {args.scene_idx} out of range (total {len(scenes)})")

    scene_name = scenes[args.scene_idx]
    scene = dump['data'][scene_name]
    obs = np.array(scene['obs_traj'])
    gt = np.array(scene['pred_traj'])
    samples = np.array(scene['llm_processed'])

    if args.ped_idx >= obs.shape[0]:
        raise IndexError(f"ped_idx {args.ped_idx} out of range (num ped {obs.shape[0]})")

    default_out = dump_dir / f'{dataset}_scene{int(scene_name.split("_")[-1]):04d}_ped{args.ped_idx}.png'
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot(obs[args.ped_idx, :, 0], obs[args.ped_idx, :, 1], 'bo-', label='observed')
    plt.plot(gt[args.ped_idx, :, 0], gt[args.ped_idx, :, 1], 'g^-', label='ground truth')
    for s in samples[args.ped_idx]:
        plt.plot(s[:, 0], s[:, 1], color='orange', alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{dataset} {scene_name} ped {args.ped_idx}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
