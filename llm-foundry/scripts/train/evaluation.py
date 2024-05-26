import torch
import json
import os
import json
import os.path
import re
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
from tqdm.auto import tqdm
from lm_eval import tasks, evaluator, utils


viggo_prompt = (
    "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. "
    "This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. "
    "The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"
    "\n\n### Target sentence:\n{target}\n\n### Meaning representation:\n")
viggo_rgx = r'### Meaning representation:\n(.*)'

sql_prompt = ("{instruct}\n\n answer: ")
sql_rgx = 'SELECT (.*)'

def viggo_batch_eval(model, tokenizer, prompt, rgx, batch):
    model_inputs = tokenizer([prompt.format(target=target) for target in batch], return_tensors="pt", padding=True).to("cuda")
    generated_ids = model.generate(**model_inputs, max_length=512, do_sample=False, num_beams=1)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    raw_results = results.copy()
    results = [re.search(rgx, result).group(1) for result in results]
    results = [result.strip() for result in results]
    return results, raw_results

@torch.no_grad()
def eval_viggo(model, tokenizer, out_folder):
    viggo_dataset = load_dataset('GEM/viggo', split='validation')
    dataloader = DataLoader(viggo_dataset, batch_size=16, shuffle=False)

    results = []
    count = 0
    with open(os.path.join(out_folder, 'viggo_write_out_info.json'), 'w') as file:
        for batch in tqdm(dataloader):
            preds, raw_preds = viggo_batch_eval(model, tokenizer, viggo_prompt, viggo_rgx, batch['target'])

            labels = batch['meaning_representation']
            for instruction, label, raw_pred, pred in zip(batch['target'], labels, raw_preds, preds):
                count += 1
                file.write(f'{count=}|@{instruction=}@\n')
                file.write(f'{count=}|@{label=}@\n')
                file.write(f'{count=}|@{raw_pred=}@\n')
                file.write(f'{count=}|@{pred=}@\n')
                file.write(f'{count=}|@correct={pred.strip() == label.strip()}@\n\n')

            results.extend([pred.strip() == label.strip() for pred, label in zip(preds, labels)])

    return sum(results) / len(results)

def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        load_in_4bit=False,
        quantization_config=None,
        trust_remote_code=True,
        use_auth_token=True
    )
    model.eval()
    return model, tokenizer

def _evaluate_custom_dataset(ft_model_path, metrics_out_folder, task):
    """
        Main method for evaluating a model on viggo dataset.
    """
    model, tokenizer = load_model(ft_model_path)

    accuracy_std = 0
    if task == 'viggo':
        accuracy = eval_viggo(model, tokenizer, metrics_out_folder)
    elif task == 'sql':
        accuracy = eval_sql(model, tokenizer, metrics_out_folder)
    else:
        raise RuntimeError(f'The dataset {task} is not in the list of custom tasks')

    with open(os.path.join(metrics_out_folder, f'{task}_eval_metrics.json'), 'w') as file:
        json.dump(dict(acc=accuracy, acc_std=accuracy_std), file)

    return accuracy, accuracy_std

def sql_batch_eval(model, tokenizer, prompt, rgx, batch):
    model_inputs = tokenizer([prompt.format(instruct=instruction) for instruction in batch], return_tensors="pt",
                             padding=True).to("cuda")
    generated_ids = model.generate(**model_inputs, max_length=512, do_sample=False, num_beams=1)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    raw_results = results.copy()
    results = [re.search(rgx, result).group(0) if re.search(rgx, result) else '' for result in results]
    results = [result.strip() for result in results]
    return results, raw_results

@torch.no_grad()
def eval_sql(model, tokenizer, out_folder):
    sql_dataset = load_dataset('json',
                               data_files="/mnt/beegfs/alistgrp/imodoran/datasets/sql/valid.jsonl",
                               split="train")
    sql_dataset = sql_dataset.map(
        lambda example: {
            'instruction': sql_prompt.format(instruct=example['messages'][0]['content']),
            'label': example['messages'][1]['content'],
        }, remove_columns=['messages'])
    dataloader = DataLoader(sql_dataset, batch_size=16, shuffle=False)

    results = []
    count = 0
    with open(os.path.join(out_folder, 'sql_write_out_info.json'), 'w') as file:
        for batch in tqdm(dataloader):
            preds, raw_preds = sql_batch_eval(model, tokenizer, sql_prompt, sql_rgx, batch['instruction'])

            labels = batch['label']
            for instruction, label, raw_pred, pred in zip(batch['instruction'], labels, raw_preds, preds):
                count += 1
                file.write(f'{count=}|@{instruction=}@\n')
                file.write(f'{count=}|@{label=}@\n')
                file.write(f'{count=}|@{raw_pred=}@\n')
                file.write(f'{count=}|@{pred=}@\n')
                file.write(f'{count=}|@correct={pred.strip() == label.strip()}@\n\n')
            results.extend([pred.strip() == label.strip() for pred, label in zip(preds, labels)])

    return sum(results) / len(results)

def evaluate_model(ft_model_path, task, num_fewshot=0, batch_size=16, limit=None):
    """
        This method uses lm-eval-harness to evaluate a model on a target task.
        For ViGGO, see the method called in the if-statement for "viggo" below.
    """
    print(f'Evaluating model {ft_model_path} on {task}')

    out = os.path.join(ft_model_path, 'eval')
    if not os.path.isdir(out):
        os.makedirs(out)

    # if task is gsm8k, set use_cache=true in generation_config.json and config.json
    if task == 'gsm8k':
        for config_name in ['generation_config.json', 'config.json']:
            json_file = os.path.join(ft_model_path, config_name)
            with open(json_file) as handle:
                data = json.load(handle)
            data['use_cache'] = True
            with open(json_file, 'w') as handle:
                json.dump(data, handle)

    if task in ['viggo', 'sql']:
        return _evaluate_custom_dataset(ft_model_path, out, task)

    results = evaluator.simple_evaluate(
        model='hf-causal-experimental',
        model_args=f'pretrained="{ft_model_path}",use_accelerate=True,dtype=bfloat16',
        tasks=utils.pattern_match(task.split(","), tasks.ALL_TASKS),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=None,
        device='cuda:0',
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        output_base_path=out
    )
    with open(os.path.join(out, f'{task}_eval_metrics.json'), "w") as f:
        f.write(json.dumps(results, indent=4))

    accuracy = results['results'][task]['acc']
    accuracy_std = results['results'][task]['acc_stderr']

    return accuracy, accuracy_std
