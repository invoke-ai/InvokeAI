import { z } from 'zod';

export const TEXT_FONT_IDS = [
  'sans',
  'serif',
  'mono',
  'rounded',
  'script',
  'humanist',
  'slab',
  'display',
  'narrow',
  'uiSerif',
] as const;
export const zTextFontId = z.enum(TEXT_FONT_IDS);
export type TextFontId = z.infer<typeof zTextFontId>;

export const TEXT_FONT_STACKS: Array<{ id: TextFontId; label: string; stack: string }> = [
  {
    id: 'sans',
    label: 'Sans',
    stack: 'system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif',
  },
  {
    id: 'serif',
    label: 'Serif',
    stack: 'Georgia,"Times New Roman",Times,serif',
  },
  {
    id: 'mono',
    label: 'Monospace',
    stack: 'ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace',
  },
  {
    id: 'rounded',
    label: 'Rounded',
    stack: '"Trebuchet MS",Verdana,"Segoe UI",sans-serif',
  },
  {
    id: 'script',
    label: 'Script',
    stack: '"Comic Sans MS","Comic Sans","Segoe UI",sans-serif',
  },
  {
    id: 'humanist',
    label: 'Humanist',
    stack: '"Gill Sans","Gill Sans MT","Trebuchet MS","Segoe UI",sans-serif',
  },
  {
    id: 'slab',
    label: 'Slab Serif',
    stack: '"Rockwell","Cambria","Georgia","Times New Roman",serif',
  },
  {
    id: 'display',
    label: 'Display',
    stack: '"Impact","Haettenschweiler","Franklin Gothic Medium",Arial,sans-serif',
  },
  {
    id: 'narrow',
    label: 'Narrow',
    stack: '"Arial Narrow","Roboto Condensed","Segoe UI",Arial,sans-serif',
  },
  {
    id: 'uiSerif',
    label: 'UI Serif',
    stack: '"Iowan Old Style","Palatino","Book Antiqua","Times New Roman",serif',
  },
];

export const TEXT_DEFAULT_FONT_ID: TextFontId = 'sans';
export const TEXT_DEFAULT_FONT_SIZE = 48;
export const TEXT_MIN_FONT_SIZE = 8;
export const TEXT_MAX_FONT_SIZE = 500;
export const TEXT_DEFAULT_LINE_HEIGHT = 1.25;
export const TEXT_MIN_LINE_HEIGHT = 1;
export const TEXT_MAX_LINE_HEIGHT = 2;
export const TEXT_RASTER_PADDING = 4;

export const TEXT_ALIGNMENTS = ['left', 'center', 'right'] as const;
export const zTextAlignment = z.enum(TEXT_ALIGNMENTS);
export type TextAlignment = z.infer<typeof zTextAlignment>;
export const TEXT_DEFAULT_ALIGNMENT: TextAlignment = 'left';

const stripQuotes = (fontName: string) => fontName.replace(/^['"]+|['"]+$/g, '');

const splitFontStack = (stack: string) => stack.split(',').map((font) => stripQuotes(font.trim()));

const isGenericFont = (fontName: string) =>
  fontName === 'serif' || fontName === 'sans-serif' || fontName === 'monospace' || fontName === 'cursive';

/**
 * Attempts to resolve the first available font in the stack. Falls back to the first entry if availability cannot be
 * determined (e.g. server-side rendering or older browsers).
 */
export const resolveAvailableFont = (stack: string): string => {
  const fontCandidates = splitFontStack(stack);
  if (typeof document === 'undefined' || !('fonts' in document) || typeof document.fonts.check !== 'function') {
    return fontCandidates[0] ?? 'sans-serif';
  }
  for (const candidate of fontCandidates) {
    if (isGenericFont(candidate)) {
      return candidate;
    }
    try {
      if (document.fonts.check(`12px ${candidate}`)) {
        return candidate;
      }
    } catch {
      // Some browsers throw on invalid fonts; ignore and continue through the stack.
      continue;
    }
  }
  return fontCandidates[0] ?? 'sans-serif';
};

export const getFontStackById = (fontId: TextFontId): string => {
  return TEXT_FONT_STACKS.find((font) => font.id === fontId)?.stack ?? TEXT_FONT_STACKS[0]?.stack ?? 'sans-serif';
};

export const getFontLabelById = (fontId: TextFontId): string => {
  return TEXT_FONT_STACKS.find((font) => font.id === fontId)?.label ?? TEXT_FONT_STACKS[0]?.label ?? 'Sans';
};
