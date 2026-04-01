#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.evaluate import create_tools
from framework.agent import Agent
from framework.llm import OpenRouterConfig


def load_hard_prompts(repo_root: Path) -> list[str]:
    eval_path = repo_root / 'evaluation' / 'data' / 'evals_hard.json'
    data = json.loads(eval_path.read_text())
    return [item['prompt'] for item in data]


def choose_prompts(prompts: list[str], count: int) -> list[tuple[int, str]]:
    chosen: list[tuple[int, str]] = []
    for idx, prompt in enumerate(prompts):
        lower = prompt.lower()
        score = 0
        for needle in [
            'on-time', 'completed', 'current', 'performing', 'legacy', 'refund',
            'hall of fame', 'rookie', 'weighted', 'comparable', 'official',
            'session', 'default rate', 'transaction value', 'delay', 'headcount'
        ]:
            if needle in lower:
                score += 1
        if score > 0:
            chosen.append((idx, prompt))
        if len(chosen) >= count:
            return chosen
    return list(enumerate(prompts[:count]))


def main() -> None:
    parser = argparse.ArgumentParser(description='Print bootstrap debug info for representative hard prompts.')
    parser.add_argument('--repo-root', default='.', help='Path to repo root')
    parser.add_argument('--count', type=int, default=10, help='Number of representative prompts to print')
    parser.add_argument('--all', action='store_true', help='Print all hard prompts instead of a representative subset')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    prompts = load_hard_prompts(repo_root)

    cfg = OpenRouterConfig(api_key='bootstrap-debug-no-network')
    agent = Agent(cfg, create_tools())

    selected = list(enumerate(prompts)) if args.all else choose_prompts(prompts, args.count)

    print(f'Repo root: {repo_root}')
    print(f'Using {len(selected)} hard prompts')
    print('=' * 80)

    for idx, prompt in selected:
        debug = agent.debug_bootstrap(prompt)
        printable = {
            'case_index': idx,
            'prompt': prompt,
            'guides_triggered': debug['guides_triggered'],
            'catalog_triggered': debug['catalog_triggered'],
            'guide_query_terms': debug['guide_query_terms'],
            'catalog_query_terms': debug['catalog_query_terms'],
            'guide_result_count': debug['guide_result_count'],
            'catalog_result_count': debug['catalog_result_count'],
            'guide_errors': debug['guide_errors'],
            'catalog_errors': debug['catalog_errors'],
            'guide_context_preview': debug['guide_context_preview'],
            'catalog_context_preview': debug['catalog_context_preview'],
            'bootstrap_context_present': debug['bootstrap_context_present'],
            'bootstrap_context_preview': debug['bootstrap_context_preview'],
        }
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        print('-' * 80)


if __name__ == '__main__':
    main()
