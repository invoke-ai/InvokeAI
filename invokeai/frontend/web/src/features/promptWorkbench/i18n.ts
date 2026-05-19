export type PromptWorkbenchTranslation = {
  key: string;
  options?: Record<string, number | string>;
};

export const tx = (key: string, options?: Record<string, number | string>): PromptWorkbenchTranslation => ({
  key,
  options,
});
