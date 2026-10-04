import type { NumberInput as ChakraNumberInput } from '@chakra-ui/react';
import type { KeyboardEvent } from 'react';

import { chakra } from '@chakra-ui/react';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { FormNumberField, FormSlider, useNumberCommit } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { useLayoutEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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

/** Logarithmic size slider plus an exact numeric field, shared by the brush and eraser. */
export const PaintSizeControl = ({
  label,
  setSize,
  size,
}: {
  label: string;
  setSize: (size: number) => void;
  size: number;
}) => {
  const numberValue = formatBrushSize(size);
  const formatPx = useCallback(() => `${numberValue}px`, [numberValue]);
  const onSlider = useCallback((position: number) => setSize(sliderPositionToBrushSize(position)), [setSize]);
  const onCommit = useNumberCommit((value) => setSize(clampBrushSize(value)));
  const onSliderKeyDownCapture = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
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

  return (
    <>
      <FormSlider
        aria-label={label}
        formatValue={formatPx}
        getAriaValueText={formatPx}
        max={BRUSH_SIZE_SLIDER_MAX}
        min={BRUSH_SIZE_SLIDER_MIN}
        step={BRUSH_SIZE_SLIDER_STEP}
        value={brushSizeToSliderPosition(size)}
        onKeyDownCapture={onSliderKeyDownCapture}
        onValueChange={onSlider}
      />
      <FormNumberField
        aria-label={label}
        max={MAX_BRUSH_SIZE}
        min={MIN_BRUSH_SIZE}
        step={0.1}
        suffix="px"
        value={numberValue}
        onValueCommit={onCommit}
      />
    </>
  );
};

const formatPercent = (value: number): string => `${Math.round(value)}%`;

/** Opacity slider plus a percent field, shared by the brush and eraser. */
export const PaintOpacityControl = ({
  opacity,
  setOpacity,
}: {
  opacity: number;
  setOpacity: (opacity: number) => void;
}) => {
  const { t } = useTranslation();
  const label = t('widgets.canvas.toolOptions.opacity');
  const percent = Math.round(opacity * 100);
  const onSlider = useCallback((value: number) => setOpacity(value / 100), [setOpacity]);
  const onNumber = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setOpacity(Math.max(0, Math.min(100, valueAsNumber)) / 100);
      }
    },
    [setOpacity]
  );
  return (
    <>
      <FormSlider
        aria-label={label}
        formatValue={formatPercent}
        max={100}
        min={0}
        value={percent}
        onValueChange={onSlider}
      />
      <FormNumberField
        aria-label={label}
        max={100}
        min={0}
        suffix="%"
        value={String(percent)}
        onValueChange={onNumber}
      />
    </>
  );
};

/** Hardness slider plus a percent field, shared by the brush and eraser. */
export const PaintHardnessControl = ({
  hardness,
  setHardness,
}: {
  hardness: number;
  setHardness: (hardness: number) => void;
}) => {
  const { t } = useTranslation();
  const label = t('widgets.canvas.toolOptions.hardness');
  const percent = Math.round(hardness * 100);
  const onSlider = useCallback((value: number) => setHardness(value / 100), [setHardness]);
  const onNumber = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setHardness(Math.max(0, Math.min(100, valueAsNumber)) / 100);
      }
    },
    [setHardness]
  );
  return (
    <>
      <FormSlider
        aria-label={label}
        formatValue={formatPercent}
        max={100}
        min={0}
        value={percent}
        onValueChange={onSlider}
      />
      <FormNumberField
        aria-label={label}
        max={100}
        min={0}
        suffix="%"
        value={String(percent)}
        onValueChange={onNumber}
      />
    </>
  );
};

/** Live stroke preview; the edge uses the stroke session's feather formula (sigma = (1 − hardness) · d / 4). */
export const PaintStrokePreview = ({
  color,
  hardness,
  opacity,
  size,
}: {
  color: string;
  hardness: number;
  opacity: number;
  size: number;
}) => {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) {
      return;
    }
    const dpr = globalThis.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    // Fit radius + curve swing + the full 3-sigma feather inside the box, so
    // softness has room to demonstrate instead of clipping at the edges.
    const swing = height * 0.14;
    const featherFactor = 1 + 1.5 * (1 - hardness);
    const drawn = Math.max(1, Math.min(size, (height - 2 * swing - 8) / featherFactor));
    const sigma = ((1 - hardness) * drawn) / 4;
    ctx.filter = sigma > 0 ? `blur(${sigma}px)` : 'none';
    ctx.globalAlpha = opacity;
    ctx.strokeStyle = color;
    ctx.lineWidth = drawn;
    ctx.lineCap = 'round';
    ctx.beginPath();
    const inset = Math.max(12, drawn / 2 + sigma * 3);
    const mid = height / 2;
    ctx.moveTo(inset, mid + swing);
    ctx.bezierCurveTo(width * 0.35, mid - swing * 2, width * 0.65, mid + swing * 2, width - inset, mid - swing);
    ctx.stroke();
    ctx.filter = 'none';
  }, [color, hardness, opacity, size]);
  return (
    <chakra.canvas
      ref={canvasRef}
      aria-label={t('widgets.canvas.toolOptions.strokePreview')}
      bg="bg.inset"
      h="28"
      role="img"
      rounded="sm"
      w="full"
    />
  );
};
