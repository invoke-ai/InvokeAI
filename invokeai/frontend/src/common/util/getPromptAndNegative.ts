import * as InvokeAI from 'app/invokeai';
import promptToString from './promptToString';

export function getPromptAndNegative(input_prompt: InvokeAI.Prompt) {
  let prompt: string = promptToString(input_prompt);
  let negativePrompt: string | null = null;

  // Matches all negative prompts, 1st capturing group is the prompt itself
  const negativePromptRegExp = new RegExp(/\[([^\][]*)]/, 'gi');

  // Grab the actual prompt matches (capturing group 1 is 1st index of match)
  const negativePromptMatches = [...prompt.matchAll(negativePromptRegExp)].map(
    (match) => match[1]
  );

  if (negativePromptMatches.length) {
    // Build the negative prompt itself
    negativePrompt = negativePromptMatches.join(' ');

    // Replace each match, including its surrounding brackets
    // Remove each pair of empty brackets
    // Trim whitespace
    negativePromptMatches.forEach((match) => {
      prompt = prompt.replace(`[${match}]`, '').replaceAll('[]', '').trim();
    });
  }

  return [prompt, negativePrompt];
}
