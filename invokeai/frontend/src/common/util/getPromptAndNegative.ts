import * as InvokeAI from 'app/invokeai';
import promptToString from './promptToString';

export function getPromptAndNegative(input_prompt: InvokeAI.Prompt) {
  let prompt: string = promptToString(input_prompt);
  let negativePrompt: string | null = null;

  const negativePromptRegExp = new RegExp(/(?<=\[)[^\][]*(?=])/, 'gi');
  const negativePromptMatches = [...prompt.matchAll(negativePromptRegExp)];

  if (negativePromptMatches && negativePromptMatches.length > 0) {
    negativePrompt = negativePromptMatches.join(', ');
    prompt = prompt
      .replaceAll(negativePromptRegExp, '')
      .replaceAll('[]', '')
      .trim();
  }

  return [prompt, negativePrompt];
}
