import type { ReactNode } from 'react';

import { Portal, Slider as ChakraSlider, Tooltip as ChakraTooltip } from '@chakra-ui/react';
import { useCallback, useMemo, useState } from 'react';

export type SliderMark = number | { label: ReactNode; value: number };

export interface SliderProps extends Omit<ChakraSlider.RootProps, 'children'> {
  /** Formats a thumb's value for its tooltip. */
  formatValue?: (value: number, index: number) => string;
  /** Marks under the track, revealed while the slider is hovered or dragged. */
  marks?: SliderMark[];
  /** Show the formatted value above each thumb while the slider is hovered or a thumb is dragged. */
  withThumbTooltip?: boolean;
}

// Marks stay mounted so the slider keeps a stable height, but only fade in
// while the slider is hovered or dragged. This is driven by the primitives' own
// DOM state: the root carries `data-dragging` for the whole gesture.
const INTERACTION_REVEAL_CSS = {
  '& [data-part="marker-group"]': {
    opacity: 0,
    transition: 'opacity var(--wb-motion-duration-fast)',
  },
  '&:hover [data-part="marker-group"], &[data-dragging] [data-part="marker-group"]': {
    opacity: 1,
  },
};

const RANGE_MARKER_CSS = {
  '& [data-part="marker"][data-state="under-value"], & [data-part="marker"][data-state="over-value"]': {
    '--marker-bg': 'white',
  },
  '& [data-part="marker"][data-state="at-value"]': {
    '--marker-bg': 'colors.bg',
  },
};

const TOOLTIP_POSITIONING = {
  gutter: 4,
  placement: 'top',
  strategy: 'fixed',
} as const;

const formatValueDefault = (value: number): string => String(value);

const SliderThumb = ({ index, isTooltipOpen, label }: { index: number; isTooltipOpen: boolean; label: string }) => (
  <ChakraTooltip.Root closeDelay={0} open={isTooltipOpen} openDelay={0} positioning={TOOLTIP_POSITIONING}>
    <ChakraTooltip.Trigger asChild>
      <ChakraSlider.Thumb index={index}>
        <ChakraSlider.HiddenInput />
      </ChakraSlider.Thumb>
    </ChakraTooltip.Trigger>
    <Portal>
      <ChakraTooltip.Positioner>
        <ChakraTooltip.Content pointerEvents="none">{label}</ChakraTooltip.Content>
      </ChakraTooltip.Positioner>
    </Portal>
  </ChakraTooltip.Root>
);

/**
 * Workbench slider. Wraps the Chakra slider primitives with the conveniences
 * the legacy (Chakra v2) slider had: a formatted value tooltip on each thumb
 * that shows while the slider is hovered and stays up through drags, and track
 * marks that reveal on hover so dense forms stay quiet until the slider is
 * being used. Renders one thumb per entry in `value`/`defaultValue`, so it
 * covers single and range sliders.
 */
export const Slider = ({
  formatValue = formatValueDefault,
  marks,
  withThumbTooltip = true,
  ...rootProps
}: SliderProps) => {
  const {
    css,
    onBlurCapture,
    onFocusCapture,
    onPointerDown,
    onPointerEnter,
    onPointerLeave,
    onPointerUp,
    onValueChangeEnd,
    ...rest
  } = rootProps;
  const values = rest.value ?? rest.defaultValue ?? [];
  const [isHovering, setIsHovering] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const rootCss = useMemo(() => {
    const baseCss = values.length > 1 ? [INTERACTION_REVEAL_CSS, RANGE_MARKER_CSS] : [INTERACTION_REVEAL_CSS];

    return css ? [...baseCss, css] : baseCss;
  }, [css, values.length]);
  const isTooltipOpen = withThumbTooltip && (isHovering || isFocused || isDragging);
  const handleBlurCapture: NonNullable<ChakraSlider.RootProps['onBlurCapture']> = useCallback(
    (event) => {
      setIsFocused(false);
      onBlurCapture?.(event);
    },
    [onBlurCapture]
  );
  const handleFocusCapture: NonNullable<ChakraSlider.RootProps['onFocusCapture']> = useCallback(
    (event) => {
      setIsFocused(true);
      onFocusCapture?.(event);
    },
    [onFocusCapture]
  );
  const handlePointerDown: NonNullable<ChakraSlider.RootProps['onPointerDown']> = useCallback(
    (event) => {
      setIsDragging(true);
      onPointerDown?.(event);
    },
    [onPointerDown]
  );
  const handlePointerEnter: NonNullable<ChakraSlider.RootProps['onPointerEnter']> = useCallback(
    (event) => {
      setIsHovering(true);
      onPointerEnter?.(event);
    },
    [onPointerEnter]
  );
  const handlePointerLeave: NonNullable<ChakraSlider.RootProps['onPointerLeave']> = useCallback(
    (event) => {
      setIsHovering(false);
      onPointerLeave?.(event);
    },
    [onPointerLeave]
  );
  const handlePointerUp: NonNullable<ChakraSlider.RootProps['onPointerUp']> = useCallback(
    (event) => {
      setIsDragging(false);
      onPointerUp?.(event);
    },
    [onPointerUp]
  );
  const handleValueChangeEnd: NonNullable<ChakraSlider.RootProps['onValueChangeEnd']> = useCallback(
    (details) => {
      setIsDragging(false);
      onValueChangeEnd?.(details);
    },
    [onValueChangeEnd]
  );

  return (
    <ChakraSlider.Root
      css={rootCss}
      {...rest}
      onBlurCapture={handleBlurCapture}
      onFocusCapture={handleFocusCapture}
      onPointerDown={handlePointerDown}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerUp={handlePointerUp}
      onValueChangeEnd={handleValueChangeEnd}
      thumbCollisionBehavior="push"
    >
      <ChakraSlider.Control>
        <ChakraSlider.Track>
          <ChakraSlider.Range />
        </ChakraSlider.Track>
        {values.map((value, index) => (
          <SliderThumb key={index} index={index} isTooltipOpen={isTooltipOpen} label={formatValue(value, index)} />
        ))}
      </ChakraSlider.Control>
      {marks?.length ? <ChakraSlider.Marks marks={marks} /> : null}
    </ChakraSlider.Root>
  );
};
