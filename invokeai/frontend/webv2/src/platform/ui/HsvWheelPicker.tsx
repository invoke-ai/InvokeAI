import type { PointerEvent as ReactPointerEvent, KeyboardEvent as ReactKeyboardEvent } from 'react';

import { Box, chakra } from '@chakra-ui/react';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { HsvColor, Point } from './hsv';

import {
  hsvToHex,
  hsvToRgb,
  hsvToTrianglePoint,
  hueToWheelAngle,
  pointToWheelAngle,
  triangleCorners,
  trianglePointToHsv,
  wheelAngleToHue,
} from './hsv';

/** Ring thickness as a fraction of the wheel radius. */
const RING_FRACTION = 0.16;
/** Gap between the ring's inner edge and the triangle's circumradius. */
const TRIANGLE_INSET_PX = 4;
const THUMB_SIZE_PX = 12;
const HUE_STEP = 2;
const HUE_STEP_LARGE = 15;
const SV_STEP = 0.02;
const SV_STEP_LARGE = 0.1;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const THUMB_FOCUS_PROPS = { outline: '2px solid {colors.accent.solid}', outlineOffset: '2px' };

/**
 * Both HSV thumbs support keyboard input; drags emit onChange and finish with onChangeEnd. Canvas shading uses the
 * same triangle math.
 */
export const HsvWheelPicker = ({
  diameterPx = 192,
  disabled = false,
  onChange,
  onChangeEnd,
  value,
}: {
  diameterPx?: number;
  disabled?: boolean;
  onChange: (next: HsvColor) => void;
  onChangeEnd?: (next: HsvColor) => void;
  value: HsvColor;
}) => {
  const { t } = useTranslation();
  const radius = diameterPx / 2;
  const ringWidth = radius * RING_FRACTION;
  const triangleRadius = radius - ringWidth - TRIANGLE_INSET_PX;
  const rootRef = useRef<HTMLDivElement>(null);
  const triangleCanvasRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<AbortController | null>(null);

  useEffect(() => () => drag.current?.abort(), []);

  // Repaint at device resolution only for hue/geometry changes; saturation/value only move the thumb.
  useEffect(() => {
    const canvas = triangleCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || triangleRadius <= 0) {
      return;
    }
    const scale = Math.min(globalThis.devicePixelRatio || 1, 2);
    const size = Math.max(1, Math.ceil(triangleRadius * 2 * scale));
    canvas.width = size;
    canvas.height = size;
    const image = ctx.createImageData(size, size);
    const { data } = image;
    const [hueCorner, whiteCorner, blackCorner] = triangleCorners(value.h, triangleRadius);
    const v0x = whiteCorner.x - hueCorner.x;
    const v0y = whiteCorner.y - hueCorner.y;
    const v1x = blackCorner.x - hueCorner.x;
    const v1y = blackCorner.y - hueCorner.y;
    const inverseDenominator = 1 / (v0x * v1y - v1x * v0y);
    // Roughly one CSS pixel of tolerance, expressed as a barycentric weight.
    const tolerance = 1.25 / triangleRadius;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const px = x / scale - triangleRadius - hueCorner.x;
        const py = y / scale - triangleRadius - hueCorner.y;
        let weightWhite = (px * v1y - v1x * py) * inverseDenominator;
        let weightBlack = (v0x * py - px * v0y) * inverseDenominator;
        let weightHue = 1 - weightWhite - weightBlack;
        if (weightHue < -tolerance || weightWhite < -tolerance || weightBlack < -tolerance) {
          continue;
        }
        weightHue = Math.max(0, weightHue);
        weightWhite = Math.max(0, weightWhite);
        weightBlack = Math.max(0, weightBlack);
        const total = weightHue + weightWhite + weightBlack;
        const v = 1 - weightBlack / total;
        const rgb = hsvToRgb({ h: value.h, s: v === 0 ? 0 : weightHue / total / v, v });
        const offset = (y * size + x) * 4;
        data[offset] = rgb.r;
        data[offset + 1] = rgb.g;
        data[offset + 2] = rgb.b;
        data[offset + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [triangleRadius, value.h]);

  const toLocalPoint = useCallback(
    (event: { clientX: number; clientY: number }): Point => {
      const bounds = rootRef.current?.getBoundingClientRect();
      if (!bounds) {
        return { x: 0, y: 0 };
      }
      return { x: event.clientX - bounds.left - radius, y: event.clientY - bounds.top - radius };
    },
    [radius]
  );

  const beginDrag = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, apply: (point: Point) => HsvColor) => {
      event.preventDefault();
      const controller = new AbortController();
      drag.current?.abort();
      drag.current = controller;
      let latest = apply(toLocalPoint(event));
      onChange(latest);
      const move = (moveEvent: PointerEvent) => {
        latest = apply(toLocalPoint(moveEvent));
        onChange(latest);
      };
      const finish = () => {
        controller.abort();
        onChangeEnd?.(latest);
      };
      window.addEventListener('pointermove', move, { signal: controller.signal });
      window.addEventListener('pointerup', finish, { signal: controller.signal });
      window.addEventListener('pointercancel', finish, { signal: controller.signal });
    },
    [onChange, onChangeEnd, toLocalPoint]
  );

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled || event.button !== 0) {
        return;
      }
      const point = toLocalPoint(event);
      const distance = Math.hypot(point.x, point.y);
      if (distance > radius) {
        return;
      }
      // Freeze the other color axes during each drag.
      if (distance >= radius - ringWidth - TRIANGLE_INSET_PX / 2) {
        beginDrag(event, (p) => ({ ...value, h: wheelAngleToHue(pointToWheelAngle(p)) }));
      } else {
        beginDrag(event, (p) => trianglePointToHsv(p, value.h, triangleRadius));
      }
    },
    [beginDrag, disabled, radius, ringWidth, toLocalPoint, triangleRadius, value]
  );

  const step = useCallback(
    (next: HsvColor) => {
      onChange(next);
      onChangeEnd?.(next);
    },
    [onChange, onChangeEnd]
  );

  const onHueKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const size = event.shiftKey ? HUE_STEP_LARGE : HUE_STEP;
      const delta =
        event.key === 'ArrowRight' || event.key === 'ArrowUp'
          ? size
          : event.key === 'ArrowLeft' || event.key === 'ArrowDown'
            ? -size
            : undefined;
      if (delta === undefined) {
        return;
      }
      event.preventDefault();
      step({ ...value, h: (value.h + delta + 360) % 360 });
    },
    [step, value]
  );

  const onTriangleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const size = event.shiftKey ? SV_STEP_LARGE : SV_STEP;
      const current = value;
      const next =
        event.key === 'ArrowRight'
          ? { ...current, s: clamp01(current.s + size) }
          : event.key === 'ArrowLeft'
            ? { ...current, s: clamp01(current.s - size) }
            : event.key === 'ArrowUp'
              ? { ...current, v: clamp01(current.v + size) }
              : event.key === 'ArrowDown'
                ? { ...current, v: clamp01(current.v - size) }
                : undefined;
      if (!next) {
        return;
      }
      event.preventDefault();
      step(next);
    },
    [step, value]
  );

  const hueAngle = hueToWheelAngle(value.h);
  const hueThumb: Point = {
    x: Math.cos(hueAngle) * (radius - ringWidth / 2),
    y: Math.sin(hueAngle) * (radius - ringWidth / 2),
  };
  const svThumb = hsvToTrianglePoint(value, triangleRadius);
  const hex = hsvToHex(value);
  const hueHex = hsvToHex({ h: value.h, s: 1, v: 1 });
  const percent = (fraction: number): number => Math.round(fraction * 100);
  const ringCss = useMemo(
    () => ({
      background: 'conic-gradient(from 90deg, red, yellow, lime, cyan, blue, magenta, red)',
      mask: `radial-gradient(closest-side, transparent calc(100% - ${ringWidth}px), black calc(100% - ${ringWidth - 1}px))`,
    }),
    [ringWidth]
  );
  const svThumbStyle = useMemo(() => ({ backgroundColor: hex }), [hex]);
  const hueThumbStyle = useMemo(() => ({ backgroundColor: hueHex }), [hueHex]);

  return (
    <Box
      ref={rootRef}
      cursor={disabled ? 'not-allowed' : 'crosshair'}
      h={`${diameterPx}px`}
      opacity={disabled ? 0.5 : 1}
      position="relative"
      touchAction="none"
      w={`${diameterPx}px`}
      onPointerDown={onPointerDown}
    >
      <Box
        aria-disabled={disabled || undefined}
        aria-label={t('common.colorPicker.hue')}
        aria-valuemax={359}
        aria-valuemin={0}
        aria-valuenow={Math.round(value.h) % 360}
        inset="0"
        position="absolute"
        role="slider"
        rounded="full"
        tabIndex={disabled ? -1 : 0}
        css={ringCss}
        _focusVisible={THUMB_FOCUS_PROPS}
        onKeyDown={onHueKeyDown}
      />
      <chakra.canvas
        ref={triangleCanvasRef}
        h={`${triangleRadius * 2}px`}
        left={`${radius - triangleRadius}px`}
        pointerEvents="none"
        position="absolute"
        top={`${radius - triangleRadius}px`}
        w={`${triangleRadius * 2}px`}
      />
      <Box
        aria-disabled={disabled || undefined}
        aria-label={t('common.colorPicker.saturationValue')}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={percent(value.v)}
        aria-valuetext={t('common.colorPicker.saturationValueText', {
          saturation: percent(value.s),
          value: percent(value.v),
        })}
        h={`${triangleRadius * 2}px`}
        left={`${radius - triangleRadius}px`}
        pointerEvents="none"
        position="absolute"
        role="slider"
        rounded="full"
        tabIndex={disabled ? -1 : 0}
        top={`${radius - triangleRadius}px`}
        w={`${triangleRadius * 2}px`}
        _focusVisible={THUMB_FOCUS_PROPS}
        onKeyDown={onTriangleKeyDown}
      />
      <Box
        border="2px solid white"
        h={`${THUMB_SIZE_PX}px`}
        left={`${radius + svThumb.x - THUMB_SIZE_PX / 2}px`}
        pointerEvents="none"
        position="absolute"
        rounded="full"
        shadow="0 0 0 1px rgba(0,0,0,0.6)"
        style={svThumbStyle}
        top={`${radius + svThumb.y - THUMB_SIZE_PX / 2}px`}
        w={`${THUMB_SIZE_PX}px`}
      />
      <Box
        border="2px solid white"
        h={`${THUMB_SIZE_PX}px`}
        left={`${radius + hueThumb.x - THUMB_SIZE_PX / 2}px`}
        pointerEvents="none"
        position="absolute"
        rounded="full"
        shadow="0 0 0 1px rgba(0,0,0,0.6)"
        style={hueThumbStyle}
        top={`${radius + hueThumb.y - THUMB_SIZE_PX / 2}px`}
        w={`${THUMB_SIZE_PX}px`}
      />
    </Box>
  );
};
