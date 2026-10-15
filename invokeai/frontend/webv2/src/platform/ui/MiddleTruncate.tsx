import { chakra, Text, type TextProps } from '@chakra-ui/react';
import { useMemo } from 'react';

const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

const DEFAULT_TAIL_GRAPHEMES = 8;

/** Split by grapheme to preserve emoji; short strings stay entirely in the head to avoid an unshrinkable tail. */
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
 * The split's flex items read as separate words to assistive technology ("S DXL Main"), so a split label hides
 * them and exposes the whole string once through generated content, which stays out of text selection and copy.
 */
const SPLIT_LABEL_CSS = {
  '&::before': {
    clip: 'rect(0 0 0 0)',
    content: 'attr(data-full-text)',
    height: '1px',
    overflow: 'hidden',
    position: 'absolute',
    whiteSpace: 'nowrap',
    width: '1px',
  },
} as const;

/**
 * For identifiers whose suffix matters. Keep white-space: pre across the head/tail boundary so the space there
 * survives; a head with no tail collapses newlines like any single-line label. Preserve full DOM text for copy and
 * accessibility.
 */
export const MiddleTruncate = ({
  css,
  tailGraphemes = DEFAULT_TAIL_GRAPHEMES,
  text,
  ...textProps
}: MiddleTruncateProps) => {
  const { head, tail } = useMemo(() => splitTextForMiddleTruncation(text, tailGraphemes), [tailGraphemes, text]);
  const labelCss = useMemo(() => (tail ? [SPLIT_LABEL_CSS, css] : css), [css, tail]);

  return (
    <Text
      css={labelCss}
      data-full-text={tail ? text : undefined}
      display="flex"
      minW="0"
      overflow="hidden"
      title={text}
      whiteSpace="nowrap"
      {...textProps}
    >
      <chakra.span
        aria-hidden={tail ? true : undefined}
        flex="0 1 auto"
        overflow="hidden"
        textOverflow="ellipsis"
        whiteSpace={tail ? 'pre' : 'nowrap'}
      >
        {head}
      </chakra.span>
      {tail ? (
        <chakra.span aria-hidden flexShrink="0" whiteSpace="pre">
          {tail}
        </chakra.span>
      ) : null}
    </Text>
  );
};
