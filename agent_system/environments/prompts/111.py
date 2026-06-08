prompt = """
You are given few useful crafting recipes to craft items in Minecraft.

Crafting commands are of the format:

"craft [target object] using [input ingredients]"

Every round I will give you an observation,
you have to respond an action based on the state
and instruction.

You can:

- "get" an object (ingredients) from the inventory or the environment
- look-up the game inventory by "inventory"
- "craft" (target) using any of the crafting commands

Your output must strictly follow this format:

Thought: your thoughts.
Action: your next action


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