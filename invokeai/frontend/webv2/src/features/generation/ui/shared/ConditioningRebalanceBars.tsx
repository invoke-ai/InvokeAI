/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Box, Flex } from '@chakra-ui/react';
import {
  adjustRebalanceWeight,
  barFillFraction,
  getRebalanceBarScale,
  KREA2_TAP_LAYERS,
  REBALANCE_NEUTRAL_WEIGHT,
  REBALANCE_WEIGHT_MIN,
  REBALANCE_WEIGHT_STEP,
  snapRebalanceWeight,
  weightFromTrackFraction,
} from '@features/generation/core/conditioningRebalance';
import { useCallback, useRef, type KeyboardEvent, type PointerEvent } from 'react';

export const REBALANCE_TRACK_HEIGHT_PX = 80;

/**
 * One continuous recessed track rather than twelve gapped wells.
 *
 * Twelve flexed columns divide a fluid width into fractional positions, so a 1px gap
 * between them lands mid-device-pixel and antialiases to a different apparent width in
 * every column — the separators visibly drift. Spacing the *fills* inside flush columns
 * puts every edge against the same background, so the bars read evenly at any width.
 */
const TRACK_PROPS = {
  bg: 'bg.inset',
  borderRadius: 'md',
  h: `${REBALANCE_TRACK_HEIGHT_PX}px`,
  overflow: 'hidden',
  position: 'relative',
  px: '1px',
  touchAction: 'none',
  w: 'full',
} as const;

/** Half the visual space between neighbouring bars; also insets the outermost pair. */
const BAR_INSET = '1px';

const COLUMN_HOVER_PROPS = { boxShadow: 'inset 0 0 0 1px {colors.border.emphasized}' } as const;
const COLUMN_FOCUS_PROPS = {
  boxShadow: 'inset 0 0 0 1px {colors.accent.solid}',
  outline: 'none',
} as const;

/** The "no change" rule sits behind the bars and must never eat a pointer event. */
const NEUTRAL_RULE_PROPS = {
  bg: 'fg.grid',
  h: '1px',
  left: 0,
  pointerEvents: 'none',
  position: 'absolute',
  right: 0,
  zIndex: 1,
} as const;

const getColumnIndex = (clientX: number, rect: DOMRect, count: number): number => {
  if (rect.width <= 0) {
    return 0;
  }

  return Math.min(count - 1, Math.max(0, Math.floor(((clientX - rect.left) / rect.width) * count)));
};

const getWeightAt = (clientY: number, rect: DOMRect, scale: number): number =>
  rect.height <= 0 ? scale : weightFromTrackFraction((clientY - rect.top) / rect.height, scale);

/**
 * Fills the columns a fast sweep skipped over, so the committed vector matches the
 * stroke the pointer actually drew rather than only its sample points.
 */
const strokeBetween = (
  weights: number[],
  fromIndex: number,
  fromWeight: number,
  toIndex: number,
  toWeight: number,
  scale: number
): void => {
  if (fromIndex === toIndex) {
    weights[toIndex] = toWeight;

    return;
  }

  const span = Math.abs(toIndex - fromIndex);
  const direction = toIndex > fromIndex ? 1 : -1;

  for (let offset = 1; offset <= span; offset += 1) {
    weights[fromIndex + offset * direction] = snapRebalanceWeight(
      fromWeight + (toWeight - fromWeight) * (offset / span),
      scale
    );
  }
};

interface ConditioningRebalanceBarsProps {
  /** The vector currently rendered, including any in-flight drag preview. */
  weights: readonly number[];
  /** Index whose readout the parent is showing; drives the emphasized column. */
  activeIndex: number | null;
  disabled?: boolean;
  /** Builds the per-bar accessible name from its 1-based tap and its encoder layer. */
  tapLabel: (tap: number, layer: number) => string;
  onActiveIndexChange: (index: number | null) => void;
  onPreview: (weights: number[] | null) => void;
  onCommit: (weights: number[]) => void;
}

/**
 * The twelve per-tap gains as draggable vertical bars.
 *
 * Pointer input is absolute — a bar jumps to wherever you point, and a horizontal sweep
 * paints across every column it passes, which is what makes drawing a curve feel direct.
 * The parent holds `onPreview`'s live vector so a drag renders without writing settings on
 * every move; `onCommit` fires once, on release.
 */
export const ConditioningRebalanceBars = ({
  activeIndex,
  disabled,
  onActiveIndexChange,
  onCommit,
  onPreview,
  tapLabel,
  weights,
}: ConditioningRebalanceBarsProps) => {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);
  const isDraggingRef = useRef(false);
  const scale = getRebalanceBarScale(weights);

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const rect = trackRef.current?.getBoundingClientRect();

      if (disabled || rect === undefined || (event.pointerType === 'mouse' && event.button !== 0)) {
        return;
      }

      event.preventDefault();

      const working = [...weights];
      const pointerSession = new AbortController();
      let lastIndex = getColumnIndex(event.clientX, rect, working.length);
      let lastWeight = getWeightAt(event.clientY, rect, scale);

      working[lastIndex] = lastWeight;
      isDraggingRef.current = true;
      onActiveIndexChange(lastIndex);
      onPreview([...working]);
      // preventDefault above suppressed the implicit focus; without this a click
      // would leave the bar unfocusable by the arrow keys that just edited it.
      barRefs.current[lastIndex]?.focus();

      const handlePointerMove = (moveEvent: globalThis.PointerEvent) => {
        const index = getColumnIndex(moveEvent.clientX, rect, working.length);
        const weight = getWeightAt(moveEvent.clientY, rect, scale);

        strokeBetween(working, lastIndex, lastWeight, index, weight, scale);
        lastIndex = index;
        lastWeight = weight;
        onActiveIndexChange(index);
        onPreview([...working]);
      };

      const handlePointerUp = () => {
        pointerSession.abort();
        isDraggingRef.current = false;
        onPreview(null);
        onCommit(working);
      };

      window.addEventListener('pointermove', handlePointerMove, { signal: pointerSession.signal });
      window.addEventListener('pointerup', handlePointerUp, { signal: pointerSession.signal });
      window.addEventListener('pointercancel', handlePointerUp, { signal: pointerSession.signal });
    },
    [disabled, scale, weights, onActiveIndexChange, onCommit, onPreview]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>, index: number) => {
      if (disabled) {
        return;
      }

      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        const nextIndex = Math.min(weights.length - 1, Math.max(0, index + (event.key === 'ArrowRight' ? 1 : -1)));

        event.preventDefault();
        onActiveIndexChange(nextIndex);
        barRefs.current[nextIndex]?.focus();

        return;
      }

      const current = weights[index] ?? REBALANCE_NEUTRAL_WEIGHT;
      const step = event.shiftKey ? REBALANCE_WEIGHT_STEP * 10 : REBALANCE_WEIGHT_STEP;
      const nextWeights: Partial<Record<string, number>> = {
        ArrowDown: adjustRebalanceWeight(current, -step, scale),
        ArrowUp: adjustRebalanceWeight(current, step, scale),
        End: scale,
        Home: REBALANCE_WEIGHT_MIN,
      };
      const next = nextWeights[event.key];

      if (next === undefined || next === current) {
        return;
      }

      event.preventDefault();

      const committed = [...weights];

      committed[index] = next;
      onCommit(committed);
    },
    [disabled, scale, weights, onActiveIndexChange, onCommit]
  );

  const handleTrackPointerLeave = useCallback(() => {
    // A drag that wanders outside the track still owns the readout until release.
    if (!isDraggingRef.current) {
      onActiveIndexChange(null);
    }
  }, [onActiveIndexChange]);

  return (
    <Flex
      align="stretch"
      ref={trackRef}
      {...TRACK_PROPS}
      onPointerDown={handlePointerDown}
      onPointerLeave={handleTrackPointerLeave}
    >
      <Box {...NEUTRAL_RULE_PROPS} bottom={`${barFillFraction(REBALANCE_NEUTRAL_WEIGHT, scale) * 100}%`} />
      {weights.map((weight, index) => {
        const tap = index + 1;
        const layer = KREA2_TAP_LAYERS[index] ?? 0;

        return (
          <Flex
            align="stretch"
            aria-disabled={disabled || undefined}
            aria-label={tapLabel(tap, layer)}
            aria-orientation="vertical"
            aria-valuemax={scale}
            aria-valuemin={REBALANCE_WEIGHT_MIN}
            aria-valuenow={weight}
            aria-valuetext={weight.toFixed(2)}
            boxShadow={index === activeIndex ? COLUMN_HOVER_PROPS.boxShadow : undefined}
            cursor={disabled ? 'default' : 'ns-resize'}
            data-active={index === activeIndex ? '' : undefined}
            data-index={index}
            flex="1"
            justify="stretch"
            key={tap}
            minW="0"
            position="relative"
            px={BAR_INSET}
            ref={(element: HTMLDivElement | null) => {
              barRefs.current[index] = element;
            }}
            role="slider"
            tabIndex={disabled ? -1 : 0}
            _focusVisible={COLUMN_FOCUS_PROPS}
            _hover={disabled ? undefined : COLUMN_HOVER_PROPS}
            onBlur={() => onActiveIndexChange(null)}
            onFocus={() => onActiveIndexChange(index)}
            onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => handleKeyDown(event, index)}
            onPointerEnter={() => onActiveIndexChange(index)}
          >
            <Box
              alignSelf="flex-end"
              // Bars at or below neutral are holding the prompt rather than pushing it;
              // the quieter fill is what makes that readable without a legend. `muted` is
              // too close to the well to see at these heights, so the step is to
              // `emphasized`.
              bg={weight > REBALANCE_NEUTRAL_WEIGHT ? 'accent.solid' : 'accent.emphasized'}
              borderRadius="1px"
              h={`${barFillFraction(weight, scale) * 100}%`}
              // A zero tap keeps a stub so every slot stays visible and hittable now that
              // the track no longer draws a well behind each one.
              minH="2px"
              w="full"
            />
          </Flex>
        );
      })}
    </Flex>
  );
};
