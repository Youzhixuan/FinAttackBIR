import argparse
import json
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# 导入项目原有的组件
from conversers import FinancialTargetLM, FinGPTTargetLM, XuanYuanTargetLM, FinR1TargetLM
from judges import ClassificationJudge
from logits_processors import load_symbol_vocab
from financial_attack_main_classification import load_attack_samples
from task_prompts import get_task_config

def run_fair_random_baseline():
    parser = argparse.ArgumentParser(description='Fair Random Symbol Baseline (Budget-Aligned)')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['flare_headlines', 'flare_fpb', 'flare_fiqasa', 'flare_cra_polish', 'flare_ma',
                                 'fintrust_fairness', 'fintrust_fairness_balanced'])
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='result/random_baseline')
    parser.add_argument('--suffix-length', type=int, default=30)
    parser.add_argument('--attacker-model-path', type=str, default='../models/Llama-3.1-8B')
    # 2026-02-06 - Added target model selection
    parser.add_argument('--target-model', type=str, default='finma',
                        choices=['finma', 'fingpt', 'xuanyuan', 'finr1'],
                        help='Target model: finma, fingpt, xuanyuan, finr1')
    cmd_args = parser.parse_args()

    # 1. 设置查询预算 (对齐主实验：3 blocks * 4 iterations * 20 streams = 240)
    # Modified: 2026-02-01 - Aligned with actual attack budget (3 blocks, not 6)
    TOTAL_BUDGET = 240 
    BATCH_SIZE = 20  # 模拟主实验的 n_streams
    NUM_BATCHES = TOTAL_BUDGET // BATCH_SIZE # 12 次批量请求

    # 2. 加载符号词表与分词器
    vocab_data = load_symbol_vocab('symbol_vocab.json')
    symbol_ids = vocab_data['symbol_token_ids']
    attacker_tokenizer = AutoTokenizer.from_pretrained(cmd_args.attacker_model_path)

    # 3. 初始化目标模型
    # 2026-02-06 - Support all target models
    model_names = {'finma': 'FinMA-7B', 'fingpt': 'FinGPT', 'xuanyuan': 'XuanYuan-6B', 'finr1': 'Fin-R1'}
    print(f"[INIT] Loading Target Model ({model_names.get(cmd_args.target_model, 'Unknown')})...")
    
    if cmd_args.target_model == "xuanyuan":
        targetLM = XuanYuanTargetLM()
    elif cmd_args.target_model == "fingpt":
        targetLM = FinGPTTargetLM()
    elif cmd_args.target_model == "finr1":
        targetLM = FinR1TargetLM()
    else:  # finma (default)
        targetLM = FinancialTargetLM()
    
    # 4. 固定在 GPU (Random Baseline 不需要切换模型，直接固定在 GPU 加速)
    print("[INFO] Pinning model to GPU for Random Baseline speedup...")
    targetLM.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    targetLM.model = targetLM.model.to(targetLM.gpu_device)

    # 5. 初始化 Judge
    # 修正：补充缺失的 judge_max_n_tokens 和 judge_temperature
    class DummyArgs:
        judge_model = "classification"
        judge_max_n_tokens = 500
        judge_temperature = 0.0
        goal = "misclassify financial text"
        target_str = ""
    judgeLM = ClassificationJudge(DummyArgs())

    # 6. 加载数据
    samples = load_attack_samples(cmd_args.task, cmd_args.n_samples, target_model=cmd_args.target_model)

    results = []
    success_count = 0
    
    print(f"\n[START] Running Fair Random Baseline (Max {TOTAL_BUDGET} queries/sample)...")
    for idx, sample in enumerate(tqdm(samples)):
        source_input = sample['source_input']
        gold_label = str(sample['gold_label'])
        # 2026-02-07 - Fixed: choices is a top-level field in attack pool, not nested under 'doc'
        choices = sample.get('choices', sample.get('doc', {}).get('choices', []))
        # 2026-02-07 - Pass answer_map for CRA-like tasks
        task_cfg = get_task_config(cmd_args.task) if cmd_args.task else {}
        answer_map = task_cfg.get('answer_map')
        judgeLM.set_gold_label(gold_label, choices, answer_map=answer_map)

        # 保持与主实验一致的截断逻辑
        if len(source_input) > 3000:
            source_input = "..." + source_input[-3000:]

        is_success = False
        final_response = ""
        final_suffix = ""
        total_queries_used = 0

        # --- 公平预算循环 (批量测试) ---
        for b in range(NUM_BATCHES):
            batch_prompts = []
            batch_suffixes = []
            
            for _ in range(BATCH_SIZE):
                random_token_ids = random.sample(symbol_ids, k=cmd_args.suffix_length)
                suffix = attacker_tokenizer.decode(random_token_ids, skip_special_tokens=True)
                batch_suffixes.append(suffix)
                batch_prompts.append(source_input + " " + suffix)
            
            responses = targetLM.get_response(batch_prompts)
            total_queries_used += BATCH_SIZE
            
            scores = judgeLM.score(batch_prompts, responses)
            
            if 10 in scores:
                win_idx = scores.index(10)
                is_success = True
                final_response = responses[win_idx]
                final_suffix = batch_suffixes[win_idx]
                break 
        
        if is_success:
            success_count += 1
            print(f"   [SUCCESS] Sample {idx} misled at query {total_queries_used}!")
        
        results.append({
            'sample_id': sample.get('index', idx),
            'success': is_success,
            'queries_used': total_queries_used,
            'final_suffix': final_suffix if is_success else "N/A",
            'prediction': final_response if is_success else "STILL_CORRECT"
        })

    # 7. 保存与统计
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    asr = success_count / len(samples)
    final_output = {
        'task': cmd_args.task,
        'target_model': cmd_args.target_model,
        'n_samples': len(samples),
        'asr': asr,
        'config': {
            'suffix_length': cmd_args.suffix_length,
            'attacker_tokenizer': cmd_args.attacker_model_path,
            'max_queries_budget': TOTAL_BUDGET
        },
        'results': results
    }
    
    # Include target model in filename
    save_path = os.path.join(cmd_args.output_dir, f"{cmd_args.target_model}_{cmd_args.task}_random_results.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Task: {cmd_args.task} | ASR: {asr:.2%}")
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    # 修正：调用正确的函数名
    run_fair_random_baseline()