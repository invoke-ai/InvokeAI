import type { NumberInput as ChakraNumberInput, SliderValueChangeDetails } from '@chakra-ui/react';

import { HStack, NumberInput } from '@chakra-ui/react';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/engineStores';
import { Slider } from '@workbench/components/ui';
import { useEraserOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

/** Same rationale as `BrushOptions`: the slider tops out below `MAX_BRUSH_SIZE`; the numeric input reaches the full range. */
const SLIDER_MAX_SIZE = 600;

const formatSizePx = (value: number): string => `${Math.round(value)}px`;
const formatOpacityPercent = (value: number): string => `${Math.round(value)}%`;

/** Eraser tool options: size (slider + numeric) and opacity. */
export const EraserOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useEraserOptions(engine);

  const sizeAriaLabel = useMemo(() => [t('widgets.canvas.toolOptions.eraserSize')], [t]);
  const opacityAriaLabel = useMemo(() => [t('widgets.canvas.toolOptions.opacity')], [t]);
  const sliderValue = useMemo(() => [Math.min(options.size, SLIDER_MAX_SIZE)], [options.size]);
  const opacityValue = useMemo(() => [Math.round(options.opacity * 100)], [options.opacity]);
  const numberInputValue = useMemo(() => String(Math.round(options.size)), [options.size]);

  const setSize = useCallback(
    (size: number) => engine.stores.eraserOptions.set({ ...options, size }),
    [engine, options]
  );

  const onSliderSizeChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next !== undefined && Number.isFinite(next)) {
        setSize(next);
      }
    },
    [setSize]
  );

  const onNumberSizeChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setSize(valueAsNumber);
      }
    },
    [setSize]
  );

  const onOpacityChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next !== undefined && Number.isFinite(next)) {
        engine.stores.eraserOptions.set({ ...options, opacity: next / 100 });
      }
    },
    [engine, options]
  );

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Slider
          aria-label={sizeAriaLabel}
          formatValue={formatSizePx}
          max={SLIDER_MAX_SIZE}
          min={MIN_BRUSH_SIZE}
          size="sm"
          value={sliderValue}
          w="7rem"
          onValueChange={onSliderSizeChange}
        />
        <NumberInput.Root
          max={MAX_BRUSH_SIZE}
          min={MIN_BRUSH_SIZE}
          size="xs"
          value={numberInputValue}
          w="4.5rem"
          onValueChange={onNumberSizeChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.eraserSize')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <Slider
        aria-label={opacityAriaLabel}
        formatValue={formatOpacityPercent}
        max={100}
        min={0}
        size="sm"
        value={opacityValue}
        w="6rem"
        onValueChange={onOpacityChange}
      />
    </HStack>
  );
};
