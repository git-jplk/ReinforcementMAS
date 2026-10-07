import argparse
from typing import Optional

from safetensors import torch


def prompt_resolve_planner(
    prompt: str,
    args: Optional[argparse.Namespace] = None,
    round_idx: int = 1,
) -> str:
    if bool(args.recursion_aware):
        max_rounds = int(args.max_rounds)
        if max_rounds == (round_idx):
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in the final round {round}.\nPlease provide an answer based on the current plan and the context of previous rounds.".format(round=round_idx, max_rounds=max_rounds)
        else:
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in round {round} of {max_rounds}.\nPlease provide your response based on the current plan and the context of previous rounds.".format(round=round_idx, max_rounds=max_rounds)
        
        prompt += "\nEnsure that if a current solution is present you always include it in your response and if possible refine it. If there is no current solution present, it is not the task of the planner to generate one."
        
    if bool(args.enable_cache):
            prompt += "\nYou have access to your previous answer. Your previous answer is the following:"
    return prompt


def prompt_resolve_refiner(
    prompt: str,
    args: Optional[argparse.Namespace] = None,
    round_idx: int = 1,
) -> str:
    if bool(args.recursion_aware):
        max_rounds = int(args.max_rounds)
        if max_rounds == (round_idx):
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in the final round {round}.\nPlease provide an answer based on the current plan and the context of previous rounds.".format(round=round_idx, max_rounds=max_rounds)
        else:
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in round {round} of {max_rounds}.\nPlease provide your response based on the current plan and the context of previous rounds.".format(round=round_idx, max_rounds=max_rounds)
        
        prompt += "\nEnsure that if a current solution is present you always include it in your response and if possible refine and critique it. If there is no current solution present, it is not the task of the refiner to generate one."
        
    if bool(args.enable_cache):
            prompt += "\nYou have access to your previous answer. Your previous answer is the following:"    
    return prompt


def prompt_resolve_solver(
    prompt: str,
    args: Optional[argparse.Namespace] = None,
    round_idx: int = 1,
) -> str:
    if bool(args.recursion_aware):
        max_rounds = int(args.max_rounds)
        if max_rounds == (round_idx):
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in the final round {round}.\nPlease provide an answer based on the current plan and the context of previous rounds. An answer must be provided.".format(round=round_idx, max_rounds=max_rounds)
        else:
            prompt += "\nBe aware that you are part of a multi-agent system that iteratively refines plans. You are currently in round {round} of {max_rounds}.\nPlease provide your response based on the current plan and the context of previous rounds.".format(round=round_idx, max_rounds=max_rounds)
        
        prompt += "\nEnsure that if a current solution is present you always include it in your response and if possible refine and critique it."
    
    if bool(args.enable_cache):
        prompt += "\nYou have access to your previous answer. Your previous answer is the following:"
        
    return prompt