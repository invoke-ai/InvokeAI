import { chakra } from '@chakra-ui/react';
import { getRebalanceSparklinePath } from '@features/generation/core/conditioningRebalance';

const SPARKLINE_VIEW_WIDTH = 100;
const SPARKLINE_VIEW_HEIGHT = 14;
const SPARKLINE_RENDER_WIDTH = 56;

const SPARKLINE_STYLE = { display: 'block', flexShrink: 0, pointerEvents: 'none' } as const;

// Resolve Chakra tokens before using them as SVG strokes.
const SparklineSvg = chakra('svg');
const SparklinePath = chakra('path');

/** Decorative: the editor supplies accessible values. */
export const ConditioningRebalanceSparkline = ({ muted, weights }: { muted?: boolean; weights: readonly number[] }) => (
  <SparklineSvg
    aria-hidden
    // Set explicit CSS dimensions to avoid numeric Chakra spacing-token interpretation.
    h={`${SPARKLINE_VIEW_HEIGHT}px`}
    preserveAspectRatio="none"
    style={SPARKLINE_STYLE}
    viewBox={`0 0 ${SPARKLINE_VIEW_WIDTH} ${SPARKLINE_VIEW_HEIGHT}`}
    w={`${SPARKLINE_RENDER_WIDTH}px`}
  >
    <SparklinePath
      d={getRebalanceSparklinePath(weights, SPARKLINE_VIEW_WIDTH, SPARKLINE_VIEW_HEIGHT)}
      fill="none"
      stroke={muted ? 'fg.muted' : 'accent.solid'}
      strokeWidth="1.5"
      vectorEffect="non-scaling-stroke"
    />
  </SparklineSvg>
);
