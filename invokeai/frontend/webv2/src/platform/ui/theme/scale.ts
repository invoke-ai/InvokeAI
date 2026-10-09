/**
 * The workbench's size scales. `md` is the working size: a control or text that names no size renders dense enough
 * for the workbench, smaller names are for tight inline spots, and larger ones for roomier surfaces.
 */

/** Control heights in px; button, input, select, and segment-group sizes of the same name align. */
export const CONTROL_HEIGHT_PX = { xs: 20, sm: 24, md: 28, lg: 32, xl: 36, '2xl': 40, '3xl': 44 } as const;

export type ControlSize = keyof typeof CONTROL_HEIGHT_PX;

/** Prompt editors share this scale through a root CSS property set by the appearance preference. */
export const PROMPT_FONT_SIZES = {
  small: '0.75rem',
  default: '0.82rem',
  large: '0.9375rem',
  larger: '1.0625rem',
} as const;

export type PromptFontSize = keyof typeof PROMPT_FONT_SIZES;

export const PROMPT_FONT_SIZE_PROPERTY = '--prompt-font-size';
export const PROMPT_FONT_SIZE = `var(${PROMPT_FONT_SIZE_PROPERTY}, ${PROMPT_FONT_SIZES.default})`;

export const isPromptFontSize = (value: unknown): value is PromptFontSize =>
  typeof value === 'string' && Object.hasOwn(PROMPT_FONT_SIZES, value);

interface TypeStep {
  fontSize: string;
  letterSpacing?: string;
  lineHeight: string;
}

/** Font size and line height per text style; `md` (12px) is body text. */
export const TYPE_SCALE = {
  xs: { fontSize: '0.625rem', lineHeight: '0.75rem' },
  sm: { fontSize: '0.6875rem', lineHeight: '0.875rem' },
  md: { fontSize: '0.75rem', lineHeight: '1rem' },
  lg: { fontSize: '0.875rem', lineHeight: '1.25rem' },
  xl: { fontSize: '1rem', lineHeight: '1.5rem' },
  '2xl': { fontSize: '1.125rem', lineHeight: '1.75rem' },
  '3xl': { fontSize: '1.25rem', lineHeight: '1.875rem' },
  '4xl': { fontSize: '1.5rem', lineHeight: '2rem' },
  '5xl': { fontSize: '1.875rem', lineHeight: '2.375rem' },
  '6xl': { fontSize: '2.25rem', letterSpacing: '-0.025em', lineHeight: '2.75rem' },
  '7xl': { fontSize: '3rem', letterSpacing: '-0.025em', lineHeight: '3.75rem' },
  '8xl': { fontSize: '3.75rem', letterSpacing: '-0.025em', lineHeight: '4.5rem' },
  '9xl': { fontSize: '4.5rem', letterSpacing: '-0.025em', lineHeight: '5.75rem' },
} as const satisfies Record<string, TypeStep>;
