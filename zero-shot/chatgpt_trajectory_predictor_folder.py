import os
import json
import time
import random
from pathlib import Path
import re
import ast

import openai
from openai import OpenAI


from tqdm import tqdm

from utils.compact_json_encoder import CompactJSONEncoder
from utils.trajectory_dataset import TrajectoryDataset

# API config
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY, base_url=os.getenv('OPENAI_API_BASE', 'https://aikey-gateway.ivia.ch/v1'))

temperature = 0.7
free_trial = False
max_timeout = 20

prompt_system = "You are a helpful assistant that Extrapolate the coordinate sequence data."
prompt_template_list = [
    "Forecast the next {1:d} (x, y) coordinates using the observed {0:d} (x, y) coordinate list.\nRespond with JSON only: a list of 5 items, each item is a list of {1:d} [x, y] pairs.\nNo words, no numbering, no newlines between lists.\nExample format: [[[x,y],...12 pairs...],[...[x,y]...],[...],[...],[...]]\n{2:s}",
    "Give me 5 more results other than the above methods.",
    "Give me 5 more results other than the above methods.",
    "Give me 5 more results other than the above methods."
]
coord_template = "({0:.2f}, {1:.2f})"
scene_name_template = "scene_{:04d}"

obs_len = 8
pred_len = 12
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent


def derive_prefix(stem: str) -> str:
    tokens = stem.split('_')
    while tokens and tokens[-1] in {"traj", "llm", "bg", "steps"}:
        tokens.pop()
    return "_".join(tokens) if tokens else stem


def process_dataset(txt_path: Path, model: str):
    dataset_name = derive_prefix(txt_path.stem)
    dump_filename = script_dir / 'output_dump' / dataset_name / f'{dataset_name}_chatgpt_api_dump.json'
    dump_filename.parent.mkdir(parents=True, exist_ok=True)

    if dump_filename.exists():
        with dump_filename.open('r') as f:
            output_dump = json.load(f)
    else:
        output_dump = {'dataset': dataset_name,
                       'llm_model': model,
                       'prompt_template': prompt_template_list,
                       'obs_len': obs_len,
                       'pred_len': pred_len,
                       'data': {}}

    test_dataset = TrajectoryDataset(str(txt_path.parent), obs_len=obs_len, pred_len=pred_len, min_ped=0, file_paths=[txt_path])
    num_scenes = len(test_dataset)
    indices = list(range(num_scenes))

    progressbar = tqdm(indices)
    progressbar.set_description(f'{dataset_name}')

    for scene_idx in indices:
        scene_name = scene_name_template.format(scene_idx)
        batch = test_dataset[scene_idx]
        obs_traj, pred_traj, non_linear_ped, _, _, _ = batch
        num_ped = obs_traj.shape[0]

        llm_response_list = [["" for _ in range(len(prompt_template_list))] for _ in range(num_ped)]
        llm_processed_list = [[] for _ in range(num_ped)]

        start_time = 0
        end_time = 0

        for ped_idx in range(num_ped):
            if scene_name in output_dump['data'].keys():
                if len(output_dump['data'][scene_name]['llm_processed']) == num_ped:
                    if len(output_dump['data'][scene_name]['llm_processed'][ped_idx]) == 20:
                        llm_response_list[ped_idx] = output_dump['data'][scene_name]['llm_output'][ped_idx]
                        llm_processed_list[ped_idx] = output_dump['data'][scene_name]['llm_processed'][ped_idx]
                        continue

            messages = [{"role": "system", "content": prompt_system}]

            for prompt_idx in range(len(prompt_template_list)):
                coord_str = '[' + ', '.join([coord_template.format(*obs_traj[ped_idx, i]) for i in range(obs_len)]) + ']'
                prompt = prompt_template_list[prompt_idx].format(obs_len, pred_len, coord_str)
                messages.append({"role": "user", "content": prompt})

                error_code = ''
                timeout = 0
                add_info = 0

                while timeout < max_timeout:
                    end_time = time.time()
                    if free_trial and start_time !=0 and end_time != 0 and end_time - start_time < 20:
                        time.sleep(20 - (end_time - start_time) + random.random() * 2)
                    start_time = time.time()

                    progressbar.set_description(f'{dataset_name} Ped {ped_idx+1}/{num_ped} Prompt {prompt_idx+1}/{len(prompt_template_list)} retry {timeout}/{max_timeout} {error_code}')

                    tmp = 1.0 if timeout >= max_timeout // 2 else temperature
                    if prompt_idx == 0 and timeout == max_timeout // 4 and add_info < 1:
                        messages[-1]['content'] += '\nProvide five hypothetical scenarios based on different extrapolation methods.'
                        add_info = 1
                    elif prompt_idx == 0 and timeout == (max_timeout // 4) * 2 and add_info < 2:
                        messages[-1]['content'] += '\nYou can use methods like linear interpolation, polynomial fitting, moving average, and more.'
                        add_info = 2

                    try:
                        response = client.chat.completions.create(model=model, messages=messages, temperature=tmp)
                        response = response.choices[0].message.content
                    except Exception as err:
                        error_code = f"Unexpected {err=}, {type(err)=}"
                        time.sleep(random.random() * 30 + 30)
                        response = ''
                        timeout += 1
                        continue

                    if 'Rate limit reached' in response:
                        error_code = 'Rate limit reached'
                        time.sleep(random.random() * 20 + 20)
                        continue
                    elif len(response) == 0:
                        timeout += 1
                        error_code = 'Empty response'
                        continue

                    try:
                        response_cleanup = re.sub('[^0-9()\[\],.\-\n ]', '', response.replace(':', '\n')).replace('(', '[').replace(')', ']')
                        parsed = []
                        for line in response_cleanup.split('\n'):
                            line = line.strip()
                            if len(line) > 20 and line.startswith('[[') and line.endswith(']]'):
                                try:
                                    parsed.append(ast.literal_eval(line))
                                except Exception:
                                    continue
                        response_cleanup = parsed

                        if (
                            len(response_cleanup) == 1
                            and isinstance(response_cleanup[0], list)
                            and len(response_cleanup[0]) >= 5
                            and all(isinstance(it, list) for it in response_cleanup[0])
                        ):
                            response_cleanup = response_cleanup[0]
                    except Exception:
                        timeout += 1
                        error_code = 'Response to list failed'
                        continue

                    if len(response_cleanup) >= 5:
                        response_cleanup = response_cleanup[-5:]

                    if (len(response_cleanup) == 5
                        and all(len(response_cleanup[i]) == pred_len for i in range(5))
                        and all(all(len(response_cleanup[i][j]) == 2 for j in range(pred_len)) for i in range(5))):
                        llm_processed_list[ped_idx].extend(response_cleanup)
                        llm_response_list[ped_idx][prompt_idx] = response
                        messages.append({"role": "assistant", "content": response})
                        break

                    else:
                        timeout += 1
                        error_code = 'Wrong response format'

                if timeout == max_timeout:
                    print('The maximum number of trials has been exceeded.')

        output_scene = {
            'num_ped': num_ped,
            'llm_output': llm_response_list,
            'llm_processed': llm_processed_list
        }
        output_dump['data'][scene_name] = output_scene
        progressbar.update(1)

    with dump_filename.open('w') as f:
        json.dump(output_dump, f, cls=CompactJSONEncoder, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process all trajectory txt files in a folder; each file becomes its own dump.")
    parser.add_argument('--data-dir', required=True, type=str, help="Directory containing trajectory txt files")
    parser.add_argument('--model', default=0, type=int, help="model id: 0 gpt-3.5-turbo-0301, 1 gpt-4-0314, 2 gpt-3.5-turbo-1106, 3 gpt-4-1106-preview, 4 azure/gpt-4o")
    args = parser.parse_args()

    model_list = ['gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'azure/gpt-4o']
    model = model_list[args.model]

    data_root = Path(args.data_dir)
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    txt_files = sorted([p for p in data_root.iterdir() if p.suffix == '.txt'])
    if not txt_files:
        print(f"No txt files found in {data_root}")
        exit(1)

    for txt in txt_files:
        process_dataset(txt, model)
