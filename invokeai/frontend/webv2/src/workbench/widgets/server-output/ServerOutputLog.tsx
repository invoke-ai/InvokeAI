import type { CSSProperties } from 'react';

import { Box, type SystemStyleObject } from '@chakra-ui/react';
import { memo, useMemo } from 'react';

import type { ServerOutputRow, TranscriptBlock } from './transcript';

import { extendedPaletteHex, segmentRow, type OutputColor, type SegmentStyle } from './rowSegments';

/** The 16 ANSI colors as theme tokens, so output stays legible in every theme; bright variants share a token. */
const ANSI_TOKENS = ['fg.subtle', 'red.fg', 'green.fg', 'yellow.fg', 'blue.fg', 'purple.fg', 'cyan.fg', 'fg.muted'];
const ANSI_BACKGROUNDS = [
  'bg.emphasized',
  'red.subtle',
  'green.subtle',
  'yellow.subtle',
  'blue.subtle',
  'purple.subtle',
  'cyan.subtle',
  'bg.muted',
];

const paletteSelectors = (attribute: 'bg' | 'fg', tokens: readonly string[], last: string): SystemStyleObject => {
  const selectors: Record<string, SystemStyleObject> = {};

  for (let index = 0; index < 16; index += 1) {
    const token = index === 15 ? last : tokens[index % 8]!;
    selectors[`& [data-${attribute}="${index}"]`] = attribute === 'fg' ? { color: token } : { background: token };
  }

  return selectors;
};

/** Every style a row can carry; nothing from the server reaches CSS except through these selectors. */
export const SERVER_OUTPUT_LOG_CSS: SystemStyleObject = {
  ...paletteSelectors('fg', ANSI_TOKENS, 'fg'),
  ...paletteSelectors('bg', ANSI_BACKGROUNDS, 'bg.muted'),
  '& [data-fg="default"]': { color: 'bg' },
  '& [data-bg="default"]': { background: 'fg' },
  '& [data-bold]': { fontWeight: '600' },
  '& [data-dim]': { opacity: 0.7 },
  '& [data-italic]': { fontStyle: 'italic' },
  '& [data-underline]': { textDecoration: 'underline' },
  // Empty rows keep their line; generated content is neither selected nor copied.
  '& [data-row]:empty::before': { content: '"\\200b"' },
  '& [data-row][data-truncated]::after': { color: 'fg.subtle', content: '" …"' },
  // Rendering is skipped only for wrapped blocks out of view. Unwrapped, every block's width counts toward how far
  // the log scrolls sideways, and a skipped block would measure zero wide.
  '&[data-wrap] > [data-block]': { contentVisibility: 'auto' },
};

const colorStyle = (color: OutputColor | undefined): string | undefined =>
  color?.kind === 'rgb'
    ? color.hex
    : color?.kind === 'palette' && color.index > 15
      ? extendedPaletteHex(color.index)
      : undefined;

const colorAttribute = (color: OutputColor | undefined): string | undefined =>
  color?.kind === 'default'
    ? 'default'
    : color?.kind === 'palette' && color.index <= 15
      ? String(color.index)
      : undefined;

const segmentProps = (style: SegmentStyle) => {
  const color = colorStyle(style.fg);
  const background = colorStyle(style.bg);

  return {
    'data-bg': colorAttribute(style.bg),
    'data-bold': style.bold ? '' : undefined,
    'data-dim': style.dim ? '' : undefined,
    'data-fg': colorAttribute(style.fg),
    'data-italic': style.italic ? '' : undefined,
    'data-underline': style.underline ? '' : undefined,
    style: color || background ? ({ background, color } satisfies CSSProperties) : undefined,
  };
};

const LogRow = ({ row }: { row: ServerOutputRow }) => (
  <div data-row={row.id} data-truncated={row.truncated ? '' : undefined}>
    {segmentRow(row).map((segment, index) =>
      segment.style ? (
        <span key={index} {...segmentProps(segment.style)}>
          {segment.text}
        </span>
      ) : (
        segment.text
      )
    )}
  </div>
);

/** A block re-renders only when a read replaced it; wrapped, its height is remembered while scrolled out of view. */
export const LogBlock = memo(function LogBlock({ block }: { block: TranscriptBlock }) {
  const rowCount = block.rows.length;
  const style = useMemo(() => ({ containIntrinsicBlockSize: `auto calc(${rowCount} * 1lh)` }), [rowCount]);

  return (
    <Box data-block={block.key} style={style}>
      {block.rows.map((row) => (
        <LogRow key={row.id} row={row} />
      ))}
    </Box>
  );
});
