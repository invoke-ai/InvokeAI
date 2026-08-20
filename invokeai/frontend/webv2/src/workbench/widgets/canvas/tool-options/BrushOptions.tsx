import type { NumberInput as ChakraNumberInput, SliderValueChangeDetails } from '@chakra-ui/react';

import { HStack, NumberInput, Text } from '@chakra-ui/react';
import { ColorPicker, Slider, ToggleIconButton } from '@platform/ui';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { useBrushOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { DropletIcon, PenLineIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

/**
 * The slider's own travel tops out well below `MAX_BRUSH_SIZE` — most brush
 * work happens under ~600px, and the numeric input still reaches the full
 * engine-clamped range for outliers.
 */
const SLIDER_MAX_SIZE = 600;

const formatSizePx = (value: number): string => `${Math.round(value)}px`;
const formatOpacityPercent = (value: number): string => `${Math.round(value)}%`;

/** Brush tool options: color swatch, size (slider + numeric), opacity, and pressure sensitivity. */
export const BrushOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useBrushOptions(engine);

  const sizeAriaLabel = useMemo(() => [t('widgets.canvas.toolOptions.brushSize')], [t]);
  const opacityAriaLabel = useMemo(() => [t('widgets.canvas.toolOptions.opacity')], [t]);
  const sliderValue = useMemo(() => [Math.min(options.size, SLIDER_MAX_SIZE)], [options.size]);
  const opacityValue = useMemo(() => [Math.round(options.opacity * 100)], [options.opacity]);
  const numberInputValue = useMemo(() => String(Math.round(options.size)), [options.size]);

  const setSize = useCallback(
    (size: number) => engine.interaction.set('brushOptions', { ...options, size }),
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
        engine.interaction.set('brushOptions', { ...options, opacity: next / 100 });
      }
    },
    [engine, options]
  );

  const onColorChange = useCallback(
    (hex: string) => engine.interaction.set('brushOptions', { ...options, color: hex }),
    [engine, options]
  );
  const sampleColor = useColorSampler(engine);

  const onPressureWidthToggle = useCallback(
    (checked: boolean) => engine.interaction.set('brushOptions', { ...options, pressureAffectsWidth: checked }),
    [engine, options]
  );

  const onPressureOpacityToggle = useCallback(
    (checked: boolean) => engine.interaction.set('brushOptions', { ...options, pressureAffectsOpacity: checked }),
    [engine, options]
  );

  return (
    <HStack align="center" gap="3">
      <ColorPicker
        aria-label={t('widgets.canvas.toolOptions.brushColor')}
        value={options.color}
        onSampleColor={sampleColor}
        onValueChange={onColorChange}
      />
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.size')}
        </Text>
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
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.brushSize')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.opacity')}
        </Text>
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
      {/*
        Two independent toggles rather than one pressure switch: width and opacity are separate
        pressure responses, and opacity additionally costs a full scratch refill per frame.
        Each is an aria-labelled button, so unlike sibling switches sharing a
        Field.Root they cannot collide on a hidden-input id.
      */}
      <ToggleIconButton
        checked={options.pressureAffectsWidth}
        icon={PenLineIcon}
        label={t('widgets.canvas.toolOptions.pressureAffectsWidth')}
        onCheckedChange={onPressureWidthToggle}
      />
      <ToggleIconButton
        checked={options.pressureAffectsOpacity}
        icon={DropletIcon}
        label={t('widgets.canvas.toolOptions.pressureAffectsOpacity')}
        onCheckedChange={onPressureOpacityToggle}
      />
    </HStack>
  );
};
