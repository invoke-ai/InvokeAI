import type { NumberInput as ChakraNumberInput, SliderValueChangeDetails } from '@chakra-ui/react';
import type { KeyboardEvent } from 'react';

import { HStack, NumberInput, Text } from '@chakra-ui/react';
import { ColorPicker, Slider, ToggleIconButton } from '@platform/ui';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { useBrushOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { DropletIcon, PenLineIcon } from 'lucide-react';
import { useCallback, useMemo, useReducer } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

export const BRUSH_SIZE_SLIDER_MAX_SIZE = 600;
export const BRUSH_SIZE_SLIDER_MIN = 0;
export const BRUSH_SIZE_SLIDER_MAX = 10_000;
/** Fine pointer resolution; keyboard changes use human-sized pixel increments below. */
export const BRUSH_SIZE_SLIDER_STEP = 1;

const LOG_SIZE_RANGE = Math.log(BRUSH_SIZE_SLIDER_MAX_SIZE / MIN_BRUSH_SIZE);

export const clampBrushSize = (value: number): number =>
  Math.max(MIN_BRUSH_SIZE, Math.min(MAX_BRUSH_SIZE, Math.round(value * 100) / 100));

export const brushSizeToSliderPosition = (size: number): number => {
  const clamped = Math.max(MIN_BRUSH_SIZE, Math.min(BRUSH_SIZE_SLIDER_MAX_SIZE, size));
  return (Math.log(clamped / MIN_BRUSH_SIZE) / LOG_SIZE_RANGE) * BRUSH_SIZE_SLIDER_MAX;
};

export const sliderPositionToBrushSize = (position: number): number => {
  const clamped = Math.max(BRUSH_SIZE_SLIDER_MIN, Math.min(BRUSH_SIZE_SLIDER_MAX, position));
  return clampBrushSize(MIN_BRUSH_SIZE * Math.exp((clamped / BRUSH_SIZE_SLIDER_MAX) * LOG_SIZE_RANGE));
};

export const formatBrushSize = (size: number): string =>
  clampBrushSize(size)
    .toFixed(2)
    .replace(/\.?0+$/, '');

const formatOpacityPercent = (value: number): string => `${Math.round(value)}%`;

export const getBrushSizeKeyboardStep = (size: number, direction: -1 | 1): number => {
  if (size < 1 || (direction < 0 && size === 1)) {
    return 0.01;
  }
  if (size < 10 || (direction < 0 && size === 10)) {
    return 0.1;
  }
  if (size < 100 || (direction < 0 && size === 100)) {
    return 1;
  }
  return 10;
};

interface PaintSizeOpacityControlsProps {
  opacity: number;
  setOpacity: (opacity: number) => void;
  setSize: (size: number) => void;
  size: number;
  sizeLabel: string;
}

/** Shared size + opacity controls for the brush and eraser. */
export const PaintSizeOpacityControls = ({
  opacity,
  setOpacity,
  setSize,
  size,
  sizeLabel,
}: PaintSizeOpacityControlsProps) => {
  const { t } = useTranslation();
  const sizeAriaLabel = useMemo(() => [sizeLabel], [sizeLabel]);
  const opacityAriaLabel = useMemo(() => [t('widgets.canvas.toolOptions.opacity')], [t]);
  const sliderValue = useMemo(() => [brushSizeToSliderPosition(size)], [size]);
  const opacityValue = useMemo(() => [Math.round(opacity * 100)], [opacity]);
  const numberInputValue = useMemo(() => formatBrushSize(size), [size]);
  const [numberInputResetVersion, resetNumberInput] = useReducer((version: number) => version + 1, 0);
  const formatCurrentSizePx = useCallback(() => `${numberInputValue}px`, [numberInputValue]);

  const onSliderSizeChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next !== undefined && Number.isFinite(next)) {
        setSize(sliderPositionToBrushSize(next));
      }
    },
    [setSize]
  );
  const onNumberSizeCommit = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setSize(clampBrushSize(valueAsNumber));
      }
      // NumberInput retains the literal draft it was given. Always remount it
      // after commit so rounding/clamping (including a no-op engine update)
      // cannot leave the field disagreeing with the accepted size.
      resetNumberInput();
    },
    [setSize]
  );
  const onSliderKeyDownCapture = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      // Preserve the slider primitive's native modifier-key semantics. This
      // override is only for unmodified logical brush-size navigation.
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
      }
      const direction = event.key === 'ArrowUp' || event.key === 'ArrowRight' || event.key === 'PageUp' ? 1 : -1;
      if (
        event.key !== 'ArrowUp' &&
        event.key !== 'ArrowRight' &&
        event.key !== 'ArrowDown' &&
        event.key !== 'ArrowLeft' &&
        event.key !== 'PageUp' &&
        event.key !== 'PageDown'
      ) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (size > BRUSH_SIZE_SLIDER_MAX_SIZE && direction > 0) {
        return;
      }
      const multiplier = event.key === 'PageUp' || event.key === 'PageDown' ? 10 : 1;
      const sliderSize = Math.min(size, BRUSH_SIZE_SLIDER_MAX_SIZE);
      setSize(
        Math.min(
          BRUSH_SIZE_SLIDER_MAX_SIZE,
          clampBrushSize(sliderSize + direction * multiplier * getBrushSizeKeyboardStep(sliderSize, direction))
        )
      );
    },
    [setSize, size]
  );
  const onOpacityChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next !== undefined && Number.isFinite(next)) {
        setOpacity(next / 100);
      }
    },
    [setOpacity]
  );

  return (
    <>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.size')}
        </Text>
        <Slider
          aria-label={sizeAriaLabel}
          formatValue={formatCurrentSizePx}
          getAriaValueText={formatCurrentSizePx}
          max={BRUSH_SIZE_SLIDER_MAX}
          min={BRUSH_SIZE_SLIDER_MIN}
          size="sm"
          step={BRUSH_SIZE_SLIDER_STEP}
          value={sliderValue}
          w="7rem"
          onKeyDownCapture={onSliderKeyDownCapture}
          onValueChange={onSliderSizeChange}
        />
        <NumberInput.Root
          max={MAX_BRUSH_SIZE}
          min={MIN_BRUSH_SIZE}
          size="xs"
          step={0.1}
          defaultValue={numberInputValue}
          key={`${numberInputValue}:${numberInputResetVersion}`}
          w="4.5rem"
          onValueCommit={onNumberSizeCommit}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={sizeLabel} fontSize="xs" />
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
    </>
  );
};

/** Brush tool options: color swatch, size (slider + numeric), opacity, and pressure sensitivity. */
export const BrushOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useBrushOptions(engine);

  const setSize = useCallback(
    (size: number) => engine.interaction.set('brushOptions', { ...options, size: clampBrushSize(size) }),
    [engine, options]
  );

  const setOpacity = useCallback(
    (opacity: number) => engine.interaction.set('brushOptions', { ...options, opacity }),
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
      <PaintSizeOpacityControls
        opacity={options.opacity}
        setOpacity={setOpacity}
        setSize={setSize}
        size={options.size}
        sizeLabel={t('widgets.canvas.toolOptions.brushSize')}
      />
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
