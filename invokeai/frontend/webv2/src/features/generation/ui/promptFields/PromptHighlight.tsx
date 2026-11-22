import type { PromptHighlightKind, PromptHighlightOptions } from '@features/generation/core/prompt/highlight';

import { Box } from '@chakra-ui/react';
import { buildPromptHighlightSegments } from '@features/generation/core/prompt/highlight';
import { useMemo } from 'react';

/** Past this, colouring costs more than it tells the reader. */
export const MAX_HIGHLIGHTED_PROMPT_LENGTH = 20_000;

export const HIGHLIGHT_STYLE_BY_KIND: Record<
  PromptHighlightKind,
  { bg?: string; color: string; textDecoration?: string; textDecorationColor?: string }
> = {
  attention: { color: 'accent.solid' },
  attentionNumeric: { color: 'fg.success' },
  // Stripped before generating, so it reads as the quietest thing on the line.
  comment: { color: 'fg.muted' },
  embedding: { bg: 'bg.warning', color: 'fg.warning' },
  error: { bg: 'bg.error', color: 'fg.error', textDecoration: 'underline wavy', textDecorationColor: 'fg.error' },
  escapedParen: { color: 'fg.muted' },
  group: { color: 'fg.subtle' },
  // Tinted per LoRA by the span's `colorPalette`.
  loraTrigger: { bg: 'colorPalette.subtle', color: 'colorPalette.fg' },
  promptFunctionArg: { bg: 'accent.subtle/20', color: 'fg' },
  promptFunctionMethod: { color: 'accent.fg' },
  promptVariable: { color: 'accent.fg' },
  // Assignment markers distinguish variable definitions from uses.
  promptVariableOperator: { color: 'fg.muted' },
  punctuation: { color: 'fg.subtle' },
  text: { color: 'fg' },
  variantBrace: { color: 'accent.fg' },
  variantRange: { color: 'fg.success' },
  variantSampler: { color: 'fg.success' },
  variantSeparator: { color: 'fg.muted' },
  variantWeight: { color: 'fg.success' },
  wildcard: { color: 'fg.subtle' },
};

/** Distinct hues, none of them red, which the prompt keeps for errors. */
const LORA_TRIGGER_PALETTES = ['blue', 'orange', 'green', 'purple', 'pink', 'teal', 'gold', 'cyan'] as const;

/** Cycles past the eighth LoRA. */
export const getLoraTriggerPalette = (colorIndex: number): string =>
  LORA_TRIGGER_PALETTES[colorIndex % LORA_TRIGGER_PALETTES.length] ?? 'blue';

export const usePromptHighlightSegments = (prompt: string, options: PromptHighlightOptions, enabled: boolean) =>
  useMemo(
    () =>
      enabled && prompt.length > 0 && prompt.length <= MAX_HIGHLIGHTED_PROMPT_LENGTH
        ? buildPromptHighlightSegments(prompt, options)
        : [],
    [enabled, options, prompt]
  );

export const PromptHighlightSpan = ({
  colorIndex,
  kind,
  text,
}: {
  colorIndex?: number;
  kind: PromptHighlightKind;
  text: string;
}) => {
  const style = HIGHLIGHT_STYLE_BY_KIND[kind];

  return (
    <Box
      as="span"
      bg={style.bg}
      colorPalette={colorIndex === undefined ? undefined : getLoraTriggerPalette(colorIndex)}
      borderRadius={style.bg ? '2px' : undefined}
      color={style.color}
      textDecoration={style.textDecoration}
      textDecorationColor={style.textDecorationColor}
    >
      {text}
    </Box>
  );
};

/** Fall back to plain text when highlighting is disabled or oversized. */
export const HighlightedPrompt = ({
  enabled = true,
  options = EMPTY_OPTIONS,
  prompt,
}: {
  prompt: string;
  enabled?: boolean;
  options?: PromptHighlightOptions;
}) => {
  const segments = usePromptHighlightSegments(prompt, options, enabled);

  if (segments.length === 0) {
    return prompt;
  }

  return segments.map((segment) => (
    <PromptHighlightSpan
      key={`${segment.range.start}:${segment.range.end}:${segment.kind}`}
      colorIndex={segment.colorIndex}
      kind={segment.kind}
      text={segment.text}
    />
  ));
};

const EMPTY_OPTIONS: PromptHighlightOptions = {};
