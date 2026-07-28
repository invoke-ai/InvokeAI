/**
 * A line-number gutter for `PromptTextarea`.
 *
 * A textarea gives no way to ask where its soft wraps fall, so the heights come
 * from a hidden mirror that lays the same text out at the same width with the
 * same metrics. Numbering follows *logical* lines: a wrapped line keeps one
 * number and the gutter entry grows to match, the way an editor behaves.
 */

import type { BoxProps } from '@chakra-ui/react';
import type { CSSProperties } from 'react';

import { Box } from '@chakra-ui/react';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';

const TABULAR_NUMS: CSSProperties = { fontVariantNumeric: 'tabular-nums' };
/** Enough for the digits plus breathing room on either side, in `ch`. */
const GUTTER_PADDING_CH = 2;

export interface PromptLineNumbersMetrics {
  fontFamily: BoxProps['fontFamily'];
  fontSize: BoxProps['fontSize'];
  lineHeight: BoxProps['lineHeight'];
  /** The textarea's border-box width, so the mirror wraps identically. */
  clientWidth: number | null;
  paddingInline: string;
  paddingBlock: string;
  scrollTop: number;
}

/** Width the textarea must reserve on its leading edge for `lineCount` numbers. */
export const getLineNumberGutterCh = (lineCount: number): number =>
  String(Math.max(lineCount, 1)).length + GUTTER_PADDING_CH;

export const PromptLineNumbers = ({ metrics, value }: { metrics: PromptLineNumbersMetrics; value: string }) => {
  const mirrorRef = useRef<HTMLDivElement | null>(null);
  const [lineHeights, setLineHeights] = useState<number[]>([]);
  const lines = useMemo(() => value.split('\n'), [value]);
  const gutterWidth = `${getLineNumberGutterCh(lines.length)}ch`;

  useLayoutEffect(() => {
    const mirror = mirrorRef.current;

    if (!mirror) {
      return;
    }

    const measure = () => {
      const heights = [...mirror.children].map((child) => child.getBoundingClientRect().height);

      setLineHeights((current) =>
        current.length === heights.length && current.every((height, index) => height === heights[index])
          ? current
          : heights
      );
    };
    const resizeObserver = new ResizeObserver(measure);

    measure();
    resizeObserver.observe(mirror);

    return () => resizeObserver.disconnect();
  }, [lines, metrics.clientWidth, metrics.fontSize, metrics.lineHeight]);

  const sharedTextStyles = {
    fontFamily: metrics.fontFamily,
    fontSize: metrics.fontSize,
    lineHeight: metrics.lineHeight,
    overflowWrap: 'break-word',
    whiteSpace: 'pre-wrap',
  } as const;

  return (
    <>
      {/* Measured, never seen. `visibility: hidden` still participates in layout,
          which is exactly what makes it measurable. */}
      <Box aria-hidden="true" inset="0" overflow="hidden" pointerEvents="none" position="absolute" visibility="hidden">
        <Box
          ref={mirrorRef}
          m="0"
          paddingBlock={metrics.paddingBlock}
          paddingInline={metrics.paddingInline}
          paddingInlineStart={`calc(${metrics.paddingInline} + ${gutterWidth})`}
          w={metrics.clientWidth ? `${metrics.clientWidth}px` : '100%'}
          {...sharedTextStyles}
        >
          {lines.map((line, index) => (
            // Zero-width space so a blank line still occupies one row.
            <Box key={index} as="div" {...sharedTextStyles}>
              {line === '' ? '​' : line}
            </Box>
          ))}
        </Box>
      </Box>

      <Box
        aria-hidden="true"
        bottom="0"
        left="0"
        overflow="hidden"
        pointerEvents="none"
        position="absolute"
        top="0"
        w={gutterWidth}
        zIndex={2}
      >
        <Box
          color="fg.subtle"
          paddingBlock={metrics.paddingBlock}
          textAlign="end"
          transform={`translateY(${-metrics.scrollTop}px)`}
          w="full"
          {...sharedTextStyles}
          paddingInlineEnd="0.75ch"
          style={TABULAR_NUMS}
        >
          {lines.map((_, index) => (
            <Box
              key={index}
              data-line-number={index + 1}
              h={lineHeights[index] ? `${lineHeights[index]}px` : undefined}
              {...sharedTextStyles}
            >
              {index + 1}
            </Box>
          ))}
        </Box>
      </Box>
    </>
  );
};
