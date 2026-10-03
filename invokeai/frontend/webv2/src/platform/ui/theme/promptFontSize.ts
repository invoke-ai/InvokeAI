/** Prompt editors' text size, chosen in Appearance; the root variable lets every prompt follow it without props. */
export const PROMPT_FONT_SIZES = {
  small: '0.75rem',
  default: '0.82rem',
  large: '0.9375rem',
  larger: '1.0625rem',
} as const;

export type PromptFontSize = keyof typeof PROMPT_FONT_SIZES;

const PROMPT_FONT_SIZE_PROPERTY = '--prompt-font-size';

/** The prompt text size as a CSS value; falls back to the default before preferences load. */
export const PROMPT_FONT_SIZE = `var(${PROMPT_FONT_SIZE_PROPERTY}, ${PROMPT_FONT_SIZES.default})`;

export const isPromptFontSize = (value: unknown): value is PromptFontSize =>
  typeof value === 'string' && Object.hasOwn(PROMPT_FONT_SIZES, value);

export const applyPromptFontSizeToRoot = (size: PromptFontSize): void => {
  document.documentElement.style.setProperty(PROMPT_FONT_SIZE_PROPERTY, PROMPT_FONT_SIZES[size]);
};
