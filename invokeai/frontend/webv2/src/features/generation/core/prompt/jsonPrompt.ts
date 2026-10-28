// Same pattern as the backend layout node: the whole prompt is one fenced block, as language models write JSON.
const PROMPT_FENCE = /^```[A-Za-z0-9_+-]*[ \t]*\r?\n([\s\S]*?)\r?\n?```$/;

/** The trimmed prompt, or the body of the one ``` block it consists of. */
export const unwrapPromptFence = (prompt: string): string => {
  const trimmed = prompt.trim();
  return PROMPT_FENCE.exec(trimmed)?.[1] ?? trimmed;
};

/**
 * Whether the prompt (unwrapped) is a JSON object or array, such as a MOSS-Image layout or a structured prompt. Its
 * braces are data, never dynamic-prompt syntax.
 */
export const isJsonPrompt = (prompt: string): boolean => {
  const body = unwrapPromptFence(prompt).trim();
  if (!body.startsWith('{') && !body.startsWith('[')) {
    return false;
  }
  try {
    const parsed: unknown = JSON.parse(body);
    return typeof parsed === 'object' && parsed !== null;
  } catch {
    return false;
  }
};
