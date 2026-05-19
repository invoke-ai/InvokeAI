const hasDynamicPromptSyntaxRegex = /\{[\s\S]*\}|__[^\r\n]+?__|\$\{[\s\S]*\}/;
export const getShouldProcessPrompt = (prompt: string): boolean => hasDynamicPromptSyntaxRegex.test(prompt);
