import { chakra } from '@chakra-ui/react';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { useLayoutEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** The track covers the sizes people paint with, on a log scale; typing and keys still reach MAX_BRUSH_SIZE. */
export const BRUSH_SIZE_TRACK_MAX = 600;
/** Sizes keep two decimals, so sub-pixel brushes stay exact. */
const BRUSH_SIZE_STEP = 0.01;

export const clampBrushSize = (value: number): number =>
  Math.max(MIN_BRUSH_SIZE, Math.min(MAX_BRUSH_SIZE, Math.round(value * 100) / 100));

export const formatBrushSize = (size: number): string =>
  clampBrushSize(size)
    .toFixed(2)
    .replace(/\.?0+$/, '');

const formatBrushSizePx = (size: number): string => `${formatBrushSize(size)}px`;

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

/** Logarithmic size scrubber shared by the brush and eraser; every step writes through, like the `[`/`]` keys. */
export const PaintSizeControl = ({
  defaultValue,
  label,
  setSize,
  size,
}: {
  /** The size double-click and Backspace restore. */
  defaultValue?: number;
  label: string;
  setSize: (size: number) => void;
  size: number;
}) => {
  const onChange = useCallback((value: number) => setSize(clampBrushSize(value)), [setSize]);
  return (
    <ScrubberField
      defaultValue={defaultValue}
      formatValue={formatBrushSizePx}
      inputMax={MAX_BRUSH_SIZE}
      label={label}
      max={BRUSH_SIZE_TRACK_MAX}
      min={MIN_BRUSH_SIZE}
      scale="log"
      step={BRUSH_SIZE_STEP}
      stepFor={getBrushSizeKeyboardStep}
      value={size}
      onChange={onChange}
    />
  );
};

const formatPercent = (value: number): string => `${value}%`;

/**
 * A 0–1 paint option (opacity, hardness) as a percent scrubber, shared by the brush and eraser. Tool options are not
 * history, so every step writes through.
 */
export const PaintPercentControl = ({
  defaultValue,
  label,
  setValue,
  value,
}: {
  /** The 0–1 value double-click and Backspace restore. */
  defaultValue: number;
  label: string;
  setValue: (value: number) => void;
  value: number;
}) => {
  const onChange = useCallback((percent: number) => setValue(percent / 100), [setValue]);
  return (
    <ScrubberField
      defaultValue={Math.round(defaultValue * 100)}
      formatValue={formatPercent}
      label={label}
      max={100}
      min={0}
      step={1}
      value={Math.round(value * 100)}
      onChange={onChange}
    />
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
