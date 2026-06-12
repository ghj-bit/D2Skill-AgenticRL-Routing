# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

TEXTCRAFT_SYSTEM_PROMPT = """You are given few useful crafting recipes to craft items in Minecraft.

Crafting commands are of the format:

"craft [target object] using [input ingredients]"

Every round I will give you an observation,
you have to respond an action based on the state
and instruction.

You can:

- "get" an object (ingredients) from the inventory or the environment
- look-up the game inventory by "inventory"
- "craft" (target) using any of the crafting commands

Reminder:

1. Always specify the quantity when using
   "get" and "craft" commands.

   Example of get:

   get 1 lapis lazuli

   Example 1 of craft:

   craft 1 blue dye using 1 lapis lazuli

   Example 2 of craft:

   craft 1 golden carrot
   using 8 gold nugget, 1 carrot

2. When using "get" command,
   do not specify whether the item comes from
   the inventory or the environment.

3. You can use ONLY crafting commands provided,
   do not use your own crafting commands.

   However, if the crafting command uses a generic
   ingredient like "planks",

   you can use special types of the same ingredient,

   e.g.

   "dark oak planks"

   in the command instead."""

TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_PREFIX_TEMPLATE = (
    'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".\n'
    'Every round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.\n'
    '{skills_prompt}'
)

TEXTCRAFT_AGENTGYM_TAGGED_OUTPUT_FORMAT = (
    'Your output must strictly follow this format:"<think>\n'
    'your thoughts.\n'
    '\n'
    '</think>\n'
    '<action>\n'
    'your next action\n'
    '</action>"\n'
    '\n'
)

TEXTCRAFT_AGENTGYM_ACTION_LABEL_OUTPUT_FORMAT = (
    'Your output must strictly follow this format:"Thought:\n'
    'your thoughts.\n'
    '\n'
    'Action:\n'
    'your next action"\n'
    '\n'
)

TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_SUFFIX = (
    'Reminder: \n'
    '1. Always specify the quantity when using "get" and "craft" commands. - Example of get: get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot\n'
    '2. When using "get" command, do not specify whether the item comes from the inventory or the environment.\n'
    '3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.\n'
    '\n'
)

TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_TEMPLATE = (
    TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_PREFIX_TEMPLATE
    + TEXTCRAFT_AGENTGYM_TAGGED_OUTPUT_FORMAT
    + TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_SUFFIX
)

TEXTCRAFT_AGENTGYM_ACTION_LABEL_INITIAL_PROMPT_TEMPLATE = (
    TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_PREFIX_TEMPLATE
    + TEXTCRAFT_AGENTGYM_ACTION_LABEL_OUTPUT_FORMAT
    + TEXTCRAFT_AGENTGYM_INITIAL_PROMPT_SUFFIX
)

TEXTCRAFT_AGENTGYM_INITIAL_PROMPT = TEXTCRAFT_AGENTGYM_ACTION_LABEL_INITIAL_PROMPT_TEMPLATE.format(skills_prompt="")

TEXTCRAFT_AGENTGYM_ASSISTANT_PROMPT = "OK. I'll follow your instructions and try my best to solve the task."

TEXTCRAFT_OUTPUT_INSTRUCTION = """You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one TextCraft action for the current step and present it within <action> </action> tags.

Output requirements:
- Output exactly two blocks and no extra text:
  <think>...</think>
  <action>...</action>
- The <action> content must be only the raw TextCraft command, with no quotes or explanation.
- Valid action forms are:
  inventory
  get [quantity] [item]
  craft [quantity] [target object] using [quantity] [input ingredient], ...
- Do not output "Thought:" or "Action:" labels."""

TEXTCRAFT_ACTION_LABEL_OUTPUT_INSTRUCTION = """Your output must strictly follow this format:
Thought:
your thoughts.

Action:
your next action

The Action content must be exactly one raw TextCraft command, with no quotes or explanation.
Valid action forms are:
  inventory
  get [quantity] [item]
  craft [quantity] [target object] using [quantity] [input ingredient], ..."""

TEXTCRAFT_TEMPLATE_NO_HIS = """
You are an expert agent playing TextCraft, a Minecraft crafting text game.

You are given useful crafting recipes. Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round you receive an observation and must respond with one action based on the state and goal.

Available action types:
- get an object from the inventory or environment, for example: get 1 lapis lazuli
- inspect your game inventory: inventory
- craft an item using one of the provided crafting commands, for example: craft 1 blue dye using 1 lapis lazuli

Important rules:
1. Always specify the quantity when using "get" and "craft" commands.
2. When using "get", do not specify whether the item comes from inventory or environment.
3. Use only crafting commands provided in the observation. If a recipe uses a generic ingredient like "planks", you may use a specific type of that ingredient, for example "dark oak planks".

Current observation:
{current_observation}

Now it's your turn to take an action.
{output_instruction}
"""

TEXTCRAFT_TEMPLATE_WITH_HISTORY = """
You are an expert agent playing TextCraft, a Minecraft crafting text game.

Task:
{task_description}

Available action types:
- get an object from the inventory or environment, for example: get 1 lapis lazuli
- inspect your game inventory: inventory
- craft an item using one of the provided crafting commands, for example: craft 1 blue dye using 1 lapis lazuli

Important rules:
1. Always specify the quantity when using "get" and "craft" commands.
2. When using "get", do not specify whether the item comes from inventory or environment.
3. Use only crafting commands provided in the observation. If a recipe uses a generic ingredient like "planks", you may use a specific type of that ingredient, for example "dark oak planks".

====================
## Current Progress
====================

You have already taken {step_count} step(s).

Recent interaction history (observation -> action):
{action_history}

Current step: {current_step}

Current observation:
{current_observation}

Now it's your turn to take an action.
{output_instruction}
"""
