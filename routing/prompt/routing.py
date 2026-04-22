ROUTING_PROMPT_TEMPLATE = """
    You are a routing agent that decides which language model to use for each step of a task.  \n
    Task: {task_description}  \
    You must choose between: \
    Description of LLM Candidates: {candidates_intro}

    ====================
    ## Current Progress
    ====================
    The router has executed {step_count} step(s) for routing the LLM.

    Previous Steps: \
    Recent interaction history:
    {action_history}

    Current step: \
    {current_step} \

    Current observation: \
    {current_observation} \

    For the next step, which model should be used?

    ### Output Format (STRICT)
    You MUST output in the following format inside <search> tags:
    <search>model_name</search>

    ### Rules:
    - model_name must exactly match one candidate (case-sensitive if provided)
    - DO NOT output anything else (no explanation, no punctuation, no extra words)
    - DO NOT include "Candidate LLM" or any prefix/suffix
    - Output exactly one tag
"""


ROUTING_PROMPT_REFINE_TEMPLATE = """
    You are a routing agent that decides which language model to use for each step of a task.  \n
    Task: {task_description}  You must choose between:

    Previous Steps:
    [If history exists, for each step i in recent history]
    Step {I} [Model: {MODEL}]:
    Action: {ACTIONPREVIEW}
    Result: {OBSERVATIONPREVIEW}

    Now, for each reasoning step:

    1. First, think inside <think> and </think> about:
        - Think which LLM is more suitable to answer your query, based on the candidate descriptions in {candidates_intro}.
    2. If you need external knowledge, write inside <search> Candidate LLM: Query </search>, where "Candidate LLM" is either SMALL or BIG, and "Query" is the question you want to ask that LLM.
    3. After each external call, the result will be returned inside <info> and </info>. You can repeat steps 1-3 as needed.
    4. When you have enough information to complete the task, provide the final answer inside <answer> and </answer> without extra explanation.
    For the next step, which model should be used? Respond with either SMALL or BIG. Decision:
"""
