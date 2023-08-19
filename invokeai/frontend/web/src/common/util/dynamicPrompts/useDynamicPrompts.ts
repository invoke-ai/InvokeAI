import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';
import { dynamicPromptsGrammar, dynamicPromptsSemantics } from './grammar';

const parsePrompt = (prompt: string): { prompts: string[]; error: boolean } => {
  const match = dynamicPromptsGrammar.match(prompt);
  if (match.failed()) {
    return { prompts: [prompt], error: true };
  }
  return { prompts: dynamicPromptsSemantics(match).expand(), error: false };
};

export const useDynamicPrompts = (prompt: string) => {
  const [debouncedPrompt] = useDebounce(prompt, 500, { leading: false });
  const result = useMemo(() => parsePrompt(debouncedPrompt), [debouncedPrompt]);
  return result;
};
