const hasOpenCloseCurlyBracesRegex = /.*\{[\s\S]*\}.*/;
export const getShouldProcessPrompt = (prompt: string): boolean => hasOpenCloseCurlyBracesRegex.test(prompt);
