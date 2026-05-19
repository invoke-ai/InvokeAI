const WILDCARD_REFERENCE_PATTERN = /__([^\r\n]+?)__/g;
const CYCLIC_WILDCARD_PATTERN = /__@[^\r\n_][^\r\n]*?__/;
const BRACED_DYNAMIC_PATTERN = /\{[\s\S]*\}|\$\{[\s\S]*\}/;

export const getHasCyclicWildcardSyntax = (prompt: string): boolean => CYCLIC_WILDCARD_PATTERN.test(prompt);

export const getHasNonCyclicDynamicPromptSyntax = (prompt: string): boolean => {
  if (BRACED_DYNAMIC_PATTERN.test(prompt)) {
    return true;
  }

  for (const match of prompt.matchAll(WILDCARD_REFERENCE_PATTERN)) {
    const reference = match[1]?.trim() ?? '';
    if (!reference.startsWith('@')) {
      return true;
    }
  }

  return false;
};

export const getHasDynamicPromptSyntax = (prompt: string): boolean =>
  getHasCyclicWildcardSyntax(prompt) || getHasNonCyclicDynamicPromptSyntax(prompt);

export const getHasMixedCyclicAndNonCyclicDynamicPromptSyntax = (prompt: string): boolean =>
  getHasCyclicWildcardSyntax(prompt) && getHasNonCyclicDynamicPromptSyntax(prompt);

export const getIsCycleOnlyDynamicPrompt = (prompt: string): boolean =>
  getHasCyclicWildcardSyntax(prompt) && !getHasNonCyclicDynamicPromptSyntax(prompt);
