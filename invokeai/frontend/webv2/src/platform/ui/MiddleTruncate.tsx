import { chakra, Text, type TextProps } from '@chakra-ui/react';
import { useMemo } from 'react';

const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

const DEFAULT_TAIL_GRAPHEMES = 8;

/**
 * Splits text so the last `tailGraphemes` graphemes can be pinned while the
 * rest ellipsizes. Splitting on graphemes rather than code units keeps
 * surrogate pairs and emoji sequences intact at the boundary. Text at or
 * under the tail length goes entirely into the head, which degrades to plain
 * end truncation instead of an unshrinkable tail wider than its container.
 */
export const splitTextForMiddleTruncation = (text: string, tailGraphemes: number): { head: string; tail: string } => {
  if (tailGraphemes <= 0) {
    return { head: text, tail: '' };
  }

  const graphemes = [...GRAPHEME_SEGMENTER.segment(text)];

  if (graphemes.length <= tailGraphemes) {
    return { head: text, tail: '' };
  }

  const splitAt = graphemes.length - tailGraphemes;

  return {
    head: graphemes
      .slice(0, splitAt)
      .map(({ segment }) => segment)
      .join(''),
    tail: graphemes
      .slice(splitAt)
      .map(({ segment }) => segment)
      .join(''),
  };
};

export interface MiddleTruncateProps extends Omit<TextProps, 'children'> {
  text: string;
  /** How many graphemes stay visible at the end when space runs out. */
  tailGraphemes?: number;
}

/**
 * Single-line text that truncates in the middle, macOS-style, so both the
 * start and the end stay readable — for names and identifiers whose
 * distinguishing part is often the suffix (file extensions, numbered copies,
 * ids). Prose and static labels should keep ordinary end truncation.
 *
 * Pure CSS: the head is a shrinkable flex item with its own ellipsis and the
 * tail never shrinks, so the split tracks container resizes without any
 * measurement. When the text fits, head and tail render seamlessly. The spans
 * use `white-space: pre` because each flex item is its own block: `nowrap`
 * would strip a space that lands at the split point, visually fusing the two
 * halves. The full text stays in the DOM, so selection, copy, and screen
 * readers all see the real string.
 */
export const MiddleTruncate = ({ tailGraphemes = DEFAULT_TAIL_GRAPHEMES, text, ...textProps }: MiddleTruncateProps) => {
  const { head, tail } = useMemo(() => splitTextForMiddleTruncation(text, tailGraphemes), [tailGraphemes, text]);

  return (
    <Text display="flex" minW="0" overflow="hidden" title={text} whiteSpace="nowrap" {...textProps}>
      <chakra.span flex="0 1 auto" overflow="hidden" textOverflow="ellipsis" whiteSpace="pre">
        {head}
      </chakra.span>
      {tail ? (
        <chakra.span flexShrink="0" whiteSpace="pre">
          {tail}
        </chakra.span>
      ) : null}
    </Text>
  );
};
