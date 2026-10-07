def build_aware_prompt(quesion: str, round: int, max_rounds: int, solver: bool) -> str:
    template_plan = quesion + "\n"
    if round == max_rounds:
        template_plan += """
            Be aware that you are part of a multi-agent system that iteratively refines plans. 
            You are currently in the final round {round}.
            Please provide an answer based on the current plan and the context of previous rounds.
        """
        
    else:
        template_plan += """
            Be aware that you are part of a multi-agent system that iteratively refines plans. 
            You are currently in round {round} of {max_rounds}.
        """
    
    template_plan += """
    Please provide your response based on the current plan and the context of previous rounds.
    Ensure that if a current solution is present you always include it in your response and if possible refine it.
    """
    if solver:
        template_plan += """
        If you are the solver, please provide the final answer based on the refined plan.
        """
    else:
        template_plan += """
            If no current solution is present,focus on the role you were assigned.
        """
        
    return template_plan.format(round=round, max_rounds=max_rounds)
    