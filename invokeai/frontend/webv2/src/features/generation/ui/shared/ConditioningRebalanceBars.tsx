/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Box, Flex, type SystemStyleObject } from '@chakra-ui/react';
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

/** Fractional-pixel gaps make separators uneven; use a continuous track. */
const TRACK_PROPS: SystemStyleObject = {
  bg: 'bg.subtle',
  borderRadius: 'md',
  borderWidth: 1,
  h: `${REBALANCE_TRACK_HEIGHT_PX}px`,
  overflow: 'hidden',
  position: 'relative',
  touchAction: 'none',
  w: 'full',
} as const;

/** Half the visual space between neighbouring bars; also insets the outermost pair. */
const BAR_INSET = '1px';

const COLUMN_HOVER_PROPS: SystemStyleObject = { boxShadow: 'inset 0 0 0 1px {colors.border.emphasized}' } as const;
const COLUMN_FOCUS_PROPS: SystemStyleObject = {
  boxShadow: 'inset 0 0 0 1px {colors.accent.solid}',
  outline: 'none',
} as const;

/** The "no change" rule sits behind the bars and must never eat a pointer event. */
const NEUTRAL_RULE_PROPS: SystemStyleObject = {
  bg: 'border',
  h: 0.5,
  insetX: 0,
  pointerEvents: 'none',
  position: 'absolute',
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

/** Interpolate columns skipped by pointer sampling. */
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

/** Emit onPreview during interaction and one onCommit on release. */
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
      // Restore focus suppressed by preventDefault so keyboard editing remains available.
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

  // CSS direction conflicts with Flex's direction prop.
  return (
    <Flex
      align="stretch"
      css={TRACK_PROPS}
      ref={trackRef}
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
            borderRadius="sm"
          >
            <Box
              alignSelf="flex-end"
              bg={weight > REBALANCE_NEUTRAL_WEIGHT ? 'accent.solid' : 'accent.emphasized'}
              borderRadius="xs"
              h={`${barFillFraction(weight, scale) * 100}%`}
              // Zero-value stubs must remain hittable without separate wells.
              minH="2px"
              w="full"
            />
          </Flex>
        );
      })}
    </Flex>
  );
};
