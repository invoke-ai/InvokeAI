import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, Stack } from '@chakra-ui/react';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { HsvColor } from './hsv';

import { hsvToHex } from './hsv';

const THUMB_SIZE_PX = 12;
const HUE_STRIP_HEIGHT_PX = 12;
const HUE_STEP = 2;
const HUE_STEP_LARGE = 15;
const SV_STEP = 0.02;
const SV_STEP_LARGE = 0.1;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const THUMB_FOCUS_PROPS = { outline: '2px solid {colors.accent.solid}', outlineOffset: '2px' };

const HUE_STRIP_CSS = {
  background: 'linear-gradient(to right, red, yellow, lime, cyan, blue, magenta, red)',
};

/** Shares HsvColor with the wheel; layered white-to-hue and transparent-to-black gradients encode saturation/value. */
export const HsvBoxPicker = ({
  disabled = false,
  heightPx = 140,
  onChange,
  onChangeEnd,
  value,
}: {
  disabled?: boolean;
  heightPx?: number;
  onChange: (next: HsvColor) => void;
  onChangeEnd?: (next: HsvColor) => void;
  value: HsvColor;
}) => {
  const { t } = useTranslation();
  const areaRef = useRef<HTMLDivElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const drag = useRef<AbortController | null>(null);

  useEffect(() => () => drag.current?.abort(), []);

  const beginDrag = useCallback(
    (
      event: ReactPointerEvent<HTMLDivElement>,
      host: HTMLDivElement | null,
      apply: (x: number, y: number) => HsvColor
    ) => {
      if (disabled || event.button !== 0 || !host) {
        return;
      }
      event.preventDefault();
      const controller = new AbortController();
      drag.current?.abort();
      drag.current = controller;
      let latest = value;
      const applyAt = (clientX: number, clientY: number): HsvColor => {
        const bounds = host.getBoundingClientRect();
        // A zero-size host mid-drag (a crushed pane) must not divide into NaN.
        if (bounds.width <= 0 || bounds.height <= 0) {
          return latest;
        }
        return apply(clamp01((clientX - bounds.left) / bounds.width), clamp01((clientY - bounds.top) / bounds.height));
      };
      latest = applyAt(event.clientX, event.clientY);
      onChange(latest);
      const move = (moveEvent: PointerEvent) => {
        latest = applyAt(moveEvent.clientX, moveEvent.clientY);
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
    [disabled, onChange, onChangeEnd, value]
  );

  const onAreaPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) =>
      beginDrag(event, areaRef.current, (x, y) => ({ ...value, s: x, v: 1 - y })),
    [beginDrag, value]
  );
  const onStripPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) =>
      beginDrag(event, stripRef.current, (x) => ({ ...value, h: x * 359.999 })),
    [beginDrag, value]
  );

  const step = useCallback(
    (next: HsvColor) => {
      onChange(next);
      onChangeEnd?.(next);
    },
    [onChange, onChangeEnd]
  );

  const onAreaKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const size = event.shiftKey ? SV_STEP_LARGE : SV_STEP;
      const next =
        event.key === 'ArrowRight'
          ? { ...value, s: clamp01(value.s + size) }
          : event.key === 'ArrowLeft'
            ? { ...value, s: clamp01(value.s - size) }
            : event.key === 'ArrowUp'
              ? { ...value, v: clamp01(value.v + size) }
              : event.key === 'ArrowDown'
                ? { ...value, v: clamp01(value.v - size) }
                : undefined;
      if (!next) {
        return;
      }
      event.preventDefault();
      step(next);
    },
    [step, value]
  );
  const onStripKeyDown = useCallback(
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

  const hex = hsvToHex(value);
  const hueHex = hsvToHex({ h: value.h, s: 1, v: 1 });
  const percent = (fraction: number): number => Math.round(fraction * 100);
  const areaCss = useMemo(
    () => ({
      background: `linear-gradient(to top, black, transparent), linear-gradient(to right, white, ${hueHex})`,
    }),
    [hueHex]
  );
  const svThumbStyle = useMemo(
    () => ({
      backgroundColor: hex,
      left: `calc(${clamp01(value.s) * 100}% - ${THUMB_SIZE_PX / 2}px)`,
      top: `calc(${(1 - clamp01(value.v)) * 100}% - ${THUMB_SIZE_PX / 2}px)`,
    }),
    [hex, value.s, value.v]
  );
  const hueThumbStyle = useMemo(
    () => ({
      backgroundColor: hueHex,
      left: `calc(${(value.h / 360) * 100}% - ${THUMB_SIZE_PX / 2}px)`,
    }),
    [hueHex, value.h]
  );

  return (
    <Stack gap="2" opacity={disabled ? 0.5 : 1} w="full">
      <Box
        ref={areaRef}
        aria-disabled={disabled || undefined}
        aria-label={t('common.colorPicker.saturationValue')}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={percent(value.v)}
        aria-valuetext={t('common.colorPicker.saturationValueText', {
          saturation: percent(value.s),
          value: percent(value.v),
        })}
        cursor={disabled ? 'not-allowed' : 'crosshair'}
        css={areaCss}
        h={`${heightPx}px`}
        position="relative"
        touchAction="none"
        role="slider"
        rounded="sm"
        tabIndex={disabled ? -1 : 0}
        w="full"
        _focusVisible={THUMB_FOCUS_PROPS}
        onKeyDown={onAreaKeyDown}
        onPointerDown={onAreaPointerDown}
      >
        <Box
          border="2px solid white"
          h={`${THUMB_SIZE_PX}px`}
          pointerEvents="none"
          position="absolute"
          rounded="full"
          shadow="0 0 0 1px rgba(0,0,0,0.6)"
          style={svThumbStyle}
          w={`${THUMB_SIZE_PX}px`}
        />
      </Box>
      <Box
        ref={stripRef}
        aria-disabled={disabled || undefined}
        aria-label={t('common.colorPicker.hue')}
        aria-valuemax={359}
        aria-valuemin={0}
        aria-valuenow={Math.round(value.h) % 360}
        cursor={disabled ? 'not-allowed' : 'ew-resize'}
        css={HUE_STRIP_CSS}
        h={`${HUE_STRIP_HEIGHT_PX}px`}
        position="relative"
        touchAction="none"
        role="slider"
        rounded="full"
        tabIndex={disabled ? -1 : 0}
        w="full"
        _focusVisible={THUMB_FOCUS_PROPS}
        onKeyDown={onStripKeyDown}
        onPointerDown={onStripPointerDown}
      >
        <Box
          border="2px solid white"
          h={`${THUMB_SIZE_PX}px`}
          pointerEvents="none"
          position="absolute"
          rounded="full"
          shadow="0 0 0 1px rgba(0,0,0,0.6)"
          style={hueThumbStyle}
          top={`${(HUE_STRIP_HEIGHT_PX - THUMB_SIZE_PX) / 2}px`}
          w={`${THUMB_SIZE_PX}px`}
        />
      </Box>
    </Stack>
  );
};
