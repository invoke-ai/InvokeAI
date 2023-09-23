import { initialDynamicPromptsState } from './dynamicPromptsSlice';

export const dynamicPromptsPersistDenylist: (keyof typeof initialDynamicPromptsState)[] =
  ['prompts'];
