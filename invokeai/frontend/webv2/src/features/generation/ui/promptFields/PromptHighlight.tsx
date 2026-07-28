/**
 * The one place prompt highlight kinds become colours.
 *
 * Shared by the prompt textarea's underlay and the dynamic prompts preview, so a
 * weight or an embedding reads the same whether you are writing the prompt or
 * looking at what it expanded into.
 */

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
  embedding: { bg: 'bg.warning', color: 'fg.warning' },
  error: { bg: 'bg.error', color: 'fg.error', textDecoration: 'underline wavy', textDecorationColor: 'fg.error' },
  escapedParen: { color: 'fg.muted' },
  group: { color: 'fg.subtle' },
  promptFunctionArg: { bg: 'accent.subtle/20', color: 'fg' },
  promptFunctionMethod: { color: 'accent.fg' },
  promptVariable: { color: 'accent.fg' },
  punctuation: { color: 'fg.subtle' },
  text: { color: 'fg' },
  variantBrace: { color: 'accent.fg' },
  variantRange: { color: 'fg.success' },
  variantSeparator: { color: 'fg.muted' },
  variantWeight: { color: 'fg.success' },
  wildcard: { color: 'fg.subtle' },
};

export const usePromptHighlightSegments = (prompt: string, options: PromptHighlightOptions, enabled: boolean) =>
  useMemo(
    () =>
      enabled && prompt.length > 0 && prompt.length <= MAX_HIGHLIGHTED_PROMPT_LENGTH
        ? buildPromptHighlightSegments(prompt, options)
        : [],
    [enabled, options, prompt]
  );

export const PromptHighlightSpan = ({ kind, text }: { kind: PromptHighlightKind; text: string }) => {
  const style = HIGHLIGHT_STYLE_BY_KIND[kind];

  return (
    <Box
      as="span"
      bg={style.bg}
      borderRadius={style.bg ? '2px' : undefined}
      color={style.color}
      textDecoration={style.textDecoration}
      textDecorationColor={style.textDecorationColor}
    >
      {text}
    </Box>
  );
};

/**
 * Renders a prompt as coloured spans, falling back to plain text when
 * highlighting is off or the prompt is too long to be worth colouring.
 */
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
      kind={segment.kind}
      text={segment.text}
    />
  ));
};

const EMPTY_OPTIONS: PromptHighlightOptions = {};
