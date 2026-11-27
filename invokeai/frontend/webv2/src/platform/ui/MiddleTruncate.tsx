import { chakra, Text, type TextProps } from '@chakra-ui/react';
import { useCallback, useMemo, useState } from 'react';

const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

const DEFAULT_TAIL_GRAPHEMES = 8;

/** Below the tail plus this much head (roughly three graphemes and the ellipsis), a middle split reads as a fragment. */
const MIN_HEAD_GRAPHEMES = 4;

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

const prefixLength = (text: string, graphemes: number): number => {
  let length = 0;
  let count = 0;

  for (const { segment } of GRAPHEME_SEGMENTER.segment(text)) {
    if (count === graphemes) {
      break;
    }
    length += segment.length;
    count += 1;
  }

  return length;
};

const textRangeWidth = (root: HTMLElement, start: number, end: number): number => {
  let offset = 0;
  let width = 0;

  for (const span of root.children) {
    const node = span.firstChild;

    if (!(node instanceof CharacterData)) {
      continue;
    }

    const from = Math.max(start, offset) - offset;
    const to = Math.min(end, offset + node.length) - offset;

    if (to > from) {
      const range = document.createRange();
      range.setStart(node, from);
      range.setEnd(node, to);
      width += range.getBoundingClientRect().width;
    }

    offset += node.length;
  }

  return width;
};

/** Range extents ignore clipping, so both forms answer alike and cannot oscillate; transforms apply to both sides. */
const fitsSplit = (
  root: HTMLElement,
  { headLength, minHeadLength, textLength }: { headLength: number; minHeadLength: number; textLength: number }
): boolean => {
  const head = root.firstElementChild;
  // The head shrinks before the root overflows, so the head reports the truncation until it is gone.
  const isTruncated = root.scrollWidth > root.clientWidth || (head !== null && head.scrollWidth > head.clientWidth);

  if (!isTruncated) {
    return true;
  }

  const needed = textRangeWidth(root, 0, minHeadLength) + textRangeWidth(root, headLength, textLength);
  const style = getComputedStyle(root);
  const scale = root.offsetWidth > 0 ? root.getBoundingClientRect().width / root.offsetWidth : 1;
  const available =
    (root.clientWidth - Number.parseFloat(style.paddingLeft) - Number.parseFloat(style.paddingRight)) * scale;

  return available + 0.5 >= needed;
};

const sharedObserverCallbacks = new WeakMap<Element, () => void>();
let sharedObserver: ResizeObserver | null = null;

/** One observer for every split label: long lists mount many of them. */
const observeResize = (node: Element, onResize: () => void): (() => void) => {
  sharedObserver ??= new ResizeObserver((entries) => {
    for (const entry of entries) {
      sharedObserverCallbacks.get(entry.target)?.();
    }
  });
  sharedObserverCallbacks.set(node, onResize);
  sharedObserver.observe(node);

  return () => {
    sharedObserverCallbacks.delete(node);
    sharedObserver?.unobserve(node);
  };
};

/** The root's ref measures the split, so callers cannot take it. */
export interface MiddleTruncateProps extends Omit<TextProps, 'children' | 'ref'> {
  text: string;
  tailGraphemes?: number;
}

/** Split flex items read as separate words, so the whole string is exposed once through generated content. */
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

/** Keeps the suffix visible; too narrow for the tail plus a few head graphemes, the head takes the whole text. */
export const MiddleTruncate = ({
  css,
  tailGraphemes = DEFAULT_TAIL_GRAPHEMES,
  text,
  ...textProps
}: MiddleTruncateProps) => {
  const { head, tail } = useMemo(() => splitTextForMiddleTruncation(text, tailGraphemes), [tailGraphemes, text]);
  const labelCss = useMemo(() => (tail ? [SPLIT_LABEL_CSS, css] : css), [css, tail]);
  const minHeadLength = useMemo(() => prefixLength(head, MIN_HEAD_GRAPHEMES), [head]);
  const [isEndTruncated, setIsEndTruncated] = useState(false);
  const endTruncated = tail !== '' && isEndTruncated;

  // Measured on attach so the first paint is right; reattaches on text change, as same-length text can differ in width.
  const rootRef = useCallback(
    (node: HTMLElement | null) => {
      if (!node || !tail) {
        return;
      }

      const lengths = { headLength: head.length, minHeadLength, textLength: text.length };
      const measure = () => setIsEndTruncated(!fitsSplit(node, lengths));
      measure();

      return observeResize(node, measure);
    },
    [head, minHeadLength, tail, text]
  );

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
      ref={rootRef}
    >
      <chakra.span
        aria-hidden={tail ? true : undefined}
        flex="0 1 auto"
        overflow="hidden"
        textOverflow="ellipsis"
        whiteSpace={tail ? 'pre' : 'nowrap'}
      >
        {endTruncated ? text : head}
      </chakra.span>
      {tail && !endTruncated ? (
        <chakra.span aria-hidden flexShrink="0" whiteSpace="pre">
          {tail}
        </chakra.span>
      ) : null}
    </Text>
  );
};
