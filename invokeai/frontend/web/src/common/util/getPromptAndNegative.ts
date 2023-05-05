import * as InvokeAI from 'app/types/invokeai';
import promptToString from './promptToString';

export function getPromptAndNegative(inputPrompt: InvokeAI.Prompt) {
  let prompt: string =
    typeof inputPrompt === 'string' ? inputPrompt : promptToString(inputPrompt);

  let negativePrompt = '';

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
