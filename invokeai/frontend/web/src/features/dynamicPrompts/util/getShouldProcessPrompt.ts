const hasOpenCloseCurlyBracesRegex = /.*\{.*\}.*/;
export const getShouldProcessPrompt = (prompt: string): boolean =>
  hasOpenCloseCurlyBracesRegex.test(prompt);
