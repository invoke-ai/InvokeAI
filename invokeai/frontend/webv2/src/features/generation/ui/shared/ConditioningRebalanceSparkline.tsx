import { chakra } from '@chakra-ui/react';
import { getRebalanceSparklinePath } from '@features/generation/core/conditioningRebalance';

const SPARKLINE_VIEW_WIDTH = 100;
const SPARKLINE_VIEW_HEIGHT = 14;
const SPARKLINE_RENDER_WIDTH = 56;

const SPARKLINE_STYLE = { display: 'block', flexShrink: 0, pointerEvents: 'none' } as const;

// Chakra-factory'd so `stroke` resolves through the theme, as the curve editor does.
// A raw `var(--chakra-colors-*)` only happens to work for palette tokens; a plain
// semantic token like `fg.muted` emits no such variable and the stroke silently
// resolves to `none`.
const SparklineSvg = chakra('svg');
const SparklinePath = chakra('path');

/**
 * The current curve at a glance, for the collapsed row. Decorative — the bar editor
 * below carries the accessible values, so this stays out of the accessibility tree.
 */
export const ConditioningRebalanceSparkline = ({ muted, weights }: { muted?: boolean; weights: readonly number[] }) => (
  <SparklineSvg
    aria-hidden
    // Sized with `h`/`w`, not the SVG `height`/`width` attributes: on a chakra-factory
    // element those are style props, and a bare `56` resolves to the spacing token
    // (14rem), not 56px.
    h={`${SPARKLINE_VIEW_HEIGHT}px`}
    preserveAspectRatio="none"
    style={SPARKLINE_STYLE}
    viewBox={`0 0 ${SPARKLINE_VIEW_WIDTH} ${SPARKLINE_VIEW_HEIGHT}`}
    w={`${SPARKLINE_RENDER_WIDTH}px`}
  >
    <SparklinePath
      d={getRebalanceSparklinePath(weights, SPARKLINE_VIEW_WIDTH, SPARKLINE_VIEW_HEIGHT)}
      fill="none"
      // Switched off, the curve is still worth reading — just not in the accent colour,
      // which is reserved for controls that are actually doing something.
      stroke={muted ? 'fg.muted' : 'accent.solid'}
      strokeWidth="1.5"
      vectorEffect="non-scaling-stroke"
    />
  </SparklineSvg>
);
