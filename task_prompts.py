"""
Task prompts and data loading for financial adversarial attacks.
Extracted from PIXIU-main/src/tasks/flare.py to remove PIXIU dependency.

Created: 2026-02-06
"""

import os
import glob
import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset

# ============================================================================
# Task Configuration
# ============================================================================

TASK_CONFIG = {
    "flare_fpb": {
        "dataset_path": "../TheFinAI/flare-fpb",
        "choices": ["positive", "negative", "neutral"],
        "judge_type": "classification",
        "lower_case": True
    },
    "flare_fiqasa": {
        "dataset_path": "../TheFinAI/flare-fiqasa",
        "choices": ["positive", "negative", "neutral"],
        "judge_type": "classification",
        "lower_case": True
    },
    "flare_headlines": {
        "dataset_path": "../TheFinAI/flare-headlines",
        "choices": ["Yes", "No"],  # gold is index: 0=Yes, 1=No
        "judge_type": "headlines_special",
        "lower_case": False
    },
    "flare_ma": {
        "dataset_path": "../TheFinAI/flare-ma",
        "choices": ["strong sell", "sell", "hold", "buy", "strong buy"],
        "judge_type": "classification",
        "lower_case": True
    },
    "flare_cra_polish": {
        "dataset_path": "../TheFinAI/cra-polish",
        "choices": ["good", "bad"],
        "answer_map": {"no": "good", "yes": "bad"},  # prompt asks yes/no, gold is good/bad
        "judge_type": "classification",
        "lower_case": True
    },
    "fintrust_fairness": {
        "dataset_path": "../FinTrust-main/fairness/personal-level_data",
        "choices": ["yes", "no"],
        "judge_type": "classification",
        "lower_case": True,
        "data_file": "fairness_personal_level_evaluation_1000_full.jsonl"
    }
}

# All supported tasks
SUPPORTED_TASKS = list(TASK_CONFIG.keys())


# ============================================================================
# Data Loading
# ============================================================================

def load_task_data(task_name: str) -> List[Dict]:
    """
    Load test data for a task.
    
    Args:
        task_name: Task name (e.g., "flare_fpb", "fintrust_fairness")
    
    Returns:
        List of document dictionaries with 'query', 'gold', 'choices' fields
    """
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Supported: {SUPPORTED_TASKS}")
    
    config = TASK_CONFIG[task_name]
    dataset_path = config["dataset_path"]
    
    # Handle FinTrust separately
    if task_name.startswith("fintrust_"):
        return _load_fintrust_data(task_name, config)
    
    # Load PIXIU parquet data
    test_file = _find_test_file(dataset_path)
    if test_file is None:
        raise FileNotFoundError(f"Cannot find test file in {dataset_path}")
    
    print(f"[INFO] Loading {task_name} from {test_file}")
    dataset = load_dataset("parquet", data_files={"test": test_file})
    test_data = dataset["test"]
    
    # Convert to list of dicts with normalized structure
    samples = []
    for i, doc in enumerate(test_data):
        sample = {
            "index": i,
            "query": doc.get("query", ""),  # All PIXIU tasks use doc["query"]
            "gold": doc.get("gold"),
            "choices": config["choices"],
            "doc": dict(doc)  # Keep original doc for reference
        }
        samples.append(sample)
    
    print(f"[INFO] Loaded {len(samples)} samples for {task_name}")
    return samples


def _find_test_file(dataset_path: str) -> Optional[str]:
    """Find test parquet file with various naming patterns."""
    possible_paths = [
        os.path.join(dataset_path, "test.parquet"),
        os.path.join(dataset_path, "data", "test-*.parquet"),
        os.path.join(dataset_path, "test-*.parquet"),
    ]
    
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def _load_fintrust_data(task_name: str, config: Dict) -> List[Dict]:
    """Load FinTrust JSONL data."""
    data_file = os.path.join(config["dataset_path"], config["data_file"])
    
    if not os.path.exists(data_file):
        # Try fallback to 100-sample file
        data_file = os.path.join(config["dataset_path"], "fairness_personal_level_evaluation_100.jsonl")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"FinTrust data file not found: {data_file}")
    
    print(f"[INFO] Loading {task_name} from {data_file}")
    
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            
            # FinTrust uses 'full_prompt' as the query
            sample = {
                "index": i,
                "query": doc.get("full_prompt", ""),
                "gold": doc.get("label", "").lower(),  # "yes" or "no"
                "choices": config["choices"],
                "doc": doc
            }
            samples.append(sample)
    
    print(f"[INFO] Loaded {len(samples)} samples for {task_name}")
    return samples


# ============================================================================
# Prompt Generation
# ============================================================================

def doc_to_text(doc: Dict, task_name: str = None) -> str:
    """
    Convert document to prompt text.
    For all PIXIU tasks, this is simply doc["query"].
    For FinTrust, this is doc["full_prompt"] (stored as "query" after loading).
    
    Args:
        doc: Document dictionary
        task_name: Optional task name (not used, kept for compatibility)
    
    Returns:
        Prompt string
    """
    return doc.get("query", "")


# ============================================================================
# Prediction Parsing and Judging
# ============================================================================

def clean_output(output: str) -> str:
    """Clean model output, remove special tokens."""
    return output.replace('</s>', '').replace('<s>', '').strip()


def _unique_label_match(text: str, labels: List[Tuple[str, str]]) -> Optional[str]:
    """
    Find a uniquely matching label in text, with substring deduplication.
    
    Args:
        text: Text to search in
        labels: List of (compare_form, return_value) tuples
    
    Returns:
        The return_value of the unique match, or None
    """
    found = [(cf, rv) for cf, rv in labels if cf in text]
    
    if len(found) > 1:
        # Remove labels that are substrings of other found labels
        # e.g., "sell" is substring of "strong sell" → keep only "strong sell"
        found = [
            (cf, rv) for cf, rv in found
            if not any(cf != other_cf and cf in other_cf for other_cf, _ in found)
        ]
    
    if len(found) == 1:
        return found[0][1]
    
    return None


def parse_prediction(response: str, choices: List[str], lower_case: bool = True,
                     answer_map: Dict[str, str] = None) -> Optional[str]:
    """
    Parse model response to extract prediction.
    Uses text-matching logic consistent with attack Judge.
    
    Improvements over basic matching:
    - Supports answer_map for tasks where prompt labels differ from gold labels
      (e.g., CRA prompt asks yes/no, gold is good/bad)
    - First-line matching to handle verbose explanatory outputs
    - Substring deduplication (e.g., "sell" vs "strong sell")
    
    Args:
        response: Model response string
        choices: List of valid choices
        lower_case: Whether to use case-insensitive matching
        answer_map: Optional dict mapping alternative answers to choice labels
                    e.g., {"no": "good", "yes": "bad"}
    
    Returns:
        Extracted prediction label or None if ambiguous
    """
    pred = clean_output(response)
    if lower_case:
        pred = pred.lower()
        choices_compare = [c.lower() for c in choices]
    else:
        choices_compare = list(choices)
    
    # Build unified label list: (compare_form, return_value)
    labels = [(c, choices[i]) for i, c in enumerate(choices_compare)]
    if answer_map:
        for key, val in answer_map.items():
            k = key.lower() if lower_case else key
            labels.append((k, val))
    
    # Step 1: Check if starts with a valid label
    for compare_form, return_val in labels:
        if pred.startswith(compare_form):
            # Guard against question-repeating pattern
            remaining = pred[len(compare_form):len(compare_form)+30]
            other_forms = [cf for cf, _ in labels if cf != compare_form]
            if any(o in remaining for o in other_forms):
                break  # Skip to step 2
            return return_val
    
    # Step 2: Check first line for unique match
    first_line = pred.split('\n')[0].strip()
    if first_line:
        result = _unique_label_match(first_line, labels)
        if result is not None:
            return result
    
    # Step 3: Check full text for unique match (with substring dedup)
    result = _unique_label_match(pred, labels)
    if result is not None:
        return result
    
    # Step 4: No unique match found
    return None


def judge_classification(doc: Dict, response: str, task_name: str) -> Tuple[bool, str, str]:
    """
    Judge if classification prediction is correct.
    
    Args:
        doc: Document dictionary with 'gold' and 'choices'
        response: Model response string
        task_name: Task name for config lookup
    
    Returns:
        (is_correct, cleaned_output, gold_label)
    """
    config = TASK_CONFIG.get(task_name, {})
    lower_case = config.get("lower_case", True)
    choices = doc.get("choices", config.get("choices", []))
    
    gold = doc.get("gold")
    # Handle integer gold (Headlines) vs string gold
    if isinstance(gold, int) and choices:
        gold_label = choices[gold]
    else:
        gold_label = str(gold)
    
    cleaned = clean_output(response)
    answer_map = config.get("answer_map")
    prediction = parse_prediction(response, choices, lower_case, answer_map=answer_map)
    
    if prediction is None:
        return False, cleaned, gold_label
    
    # Compare with proper case handling
    if lower_case:
        is_correct = prediction.lower() == gold_label.lower()
    else:
        is_correct = prediction == gold_label
    
    return is_correct, cleaned, gold_label


def judge_headlines(doc: Dict, response: str) -> Tuple[bool, str, str]:
    """
    Judge Headlines task with special logic.
    Headlines: pred = int(cleaned != "Yes")
    gold is integer index: 0=Yes, 1=No
    
    Args:
        doc: Document dictionary
        response: Model response string
    
    Returns:
        (is_correct, cleaned_output, gold_label_text)
    """
    gold_index = doc.get("gold")
    choices = doc.get("choices", ["Yes", "No"])
    gold_label_text = choices[gold_index] if gold_index < len(choices) else str(gold_index)
    
    cleaned = clean_output(response)
    
    # Headlines special logic
    pred_index = int(cleaned != "Yes")
    is_correct = (pred_index == gold_index)
    
    return is_correct, cleaned, gold_label_text


def judge_task(doc: Dict, response: str, task_name: str) -> Tuple[bool, str, str]:
    """
    Universal judging function that dispatches to correct judge.
    
    Args:
        doc: Document dictionary
        response: Model response string
        task_name: Task name
    
    Returns:
        (is_correct, cleaned_output, gold_label)
    """
    config = TASK_CONFIG.get(task_name, {})
    judge_type = config.get("judge_type", "classification")
    
    if judge_type == "headlines_special":
        return judge_headlines(doc, response)
    else:
        return judge_classification(doc, response, task_name)


# ============================================================================
# Utility Functions
# ============================================================================

def get_task_choices(task_name: str) -> List[str]:
    """Get the choices list for a task."""
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_CONFIG[task_name]["choices"]


def get_task_config(task_name: str) -> Dict:
    """Get full config for a task."""
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_CONFIG[task_name].copy()


if __name__ == "__main__":
    # Quick test
    print("Supported tasks:", SUPPORTED_TASKS)
    
    for task in SUPPORTED_TASKS[:2]:  # Test first 2 tasks
        print(f"\n--- Testing {task} ---")
        config = get_task_config(task)
        print(f"Choices: {config['choices']}")
        print(f"Judge type: {config['judge_type']}")
