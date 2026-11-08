# Dynamic prompts

The backend API has a route to process a prompt into a list of prompts using the https://github.com/adieyal/dynamicprompts syntax

In the UI, we watch the current positive prompt field for changes (debounced) and hit that route.

When generating, we queue up a graph for each of the output prompts.

There is a modal to show the list of generated prompts with a couple settings for prompt generation.

The output prompts are stored in the redux slice for ease of consumption during graph building, but only the settings are persisted across page loads. Prompts are ephemeral.
