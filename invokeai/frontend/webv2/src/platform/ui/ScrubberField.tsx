import type { StackProps } from '@chakra-ui/react';
import type { FeatureHintId } from '@platform/ui/hints';
import type {
  ChangeEvent,
  FocusEvent,
  KeyboardEvent,
  PointerEvent as ReactPointerEvent,
  MouseEvent as ReactMouseEvent,
} from 'react';

import { Box, chakra, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { INSIDE_LABEL_INSET } from '@platform/ui/Field';
import { FEATURE_HINT_FIELD_PROPS, FeatureHint } from '@platform/ui/hints';
import { fieldLabelRecipe, scrubberInteraction } from '@theme/recipes';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** The label and value sit 12px in, so an end thumb lands a text clearance outside them. */
const TRACK_INSET_PX = 6;
/** Just enough for the fill's rounded end to frame the 2px thumb. */
const FILL_OVERHANG_PX = 4;
const FILL_INSET_PX = 1.5;
const FINE_DRAG_RATIO = 0.1;
const COARSE_STEP_MULTIPLIER = 10;
const MARK_EPSILON_RATIO = 1e-6;
const TEXT_CLEARANCE_PX = 6;
const SCRUBBER_TEXT_SHADOW = '0 1px 2px {colors.bg.panel}, 0 0 4px {colors.bg.panel}';
/** A touch must travel this far horizontally before it scrubs; until then the panel may pan. */
const TOUCH_INTENT_PX = 8;

type ScrubberMarkState = 'at-value' | 'over-value' | 'under-value';

/** A drag or held-key run of `onChange` calls; settling ends it and reports its final value once. */
type Gesture = { readonly kind: 'drag' | 'keys'; latest: number; settle: () => void };

const STEP_KEYS = new Set(['ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'End', 'Home', 'PageDown', 'PageUp']);

/** Where the label ends and the value begins, in px from the frame's left edge. */
type TextExtents = { labelEnd: number; valueStart: number; width: number };

type OwnProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  /** Alt-drag snaps to these; `defaultValue` is always a stop, and stops at `min`/`max` are not drawn. */
  marks?: number[];
  /** Looser clamps for typed and keyboard values (the track bounds apply to scrubbing). */
  inputMin?: number;
  inputMax?: number;
  /** `value`, typing, keys and bounds stay in real units; `log` needs `min > 0`, else the track stays linear. */
  scale?: 'linear' | 'log';
  stepFor?: (value: number, direction: 1 | -1) => number;
  defaultValue?: number;
  disabled?: boolean;
  hint?: FeatureHintId;
  error?: string | null;
  helpText?: string;
  /** Formats the displayed value and its `aria-valuetext`; typing always edits the raw number. */
  formatValue?: (value: number) => string;
  /** Fires per step of a gesture. A drag keeps the handler it started with, so patch-style handlers only. */
  onChange: (value: number) => void;
  /** Once per gesture that called `onChange`, so previews can record one undo step; a drag keeps its first handler. */
  onChangeEnd?: (value: number) => void;
};

export type ScrubberFieldProps = OwnProps & Omit<StackProps, keyof OwnProps | 'children' | 'defaultValue' | 'onChange'>;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const countDecimals = (value: number): number => {
  const text = String(value);
  const dot = text.indexOf('.');

  return dot === -1 ? 0 : text.length - dot - 1;
};

const roundTo = (value: number, decimals: number): number => Number(value.toFixed(decimals));

const nearestMark = (rawTrack: number, marks: readonly number[], toTrack: (value: number) => number): number =>
  marks.reduce((best, mark) => (Math.abs(toTrack(mark) - rawTrack) < Math.abs(toTrack(best) - rawTrack) ? mark : best));

const isValueTarget = (target: EventTarget | null): boolean =>
  target instanceof Element && target.closest('[data-part="value"]') !== null;

const ROOT_CSS = {
  ...scrubberInteraction,
  alignItems: 'center',
  borderColor: 'border',
  borderRadius: 'control',
  borderWidth: '1px',
  cursor: 'ew-resize',
  display: 'flex',
  h: 'control.md',
  minW: '0',
  overflow: 'hidden',
  position: 'relative',
  textStyle: 'md',
  touchAction: 'pan-y',
  userSelect: 'none',
  w: 'full',
  _disabled: { cursor: 'not-allowed', opacity: 0.5 },
  '&[data-dragging]': { borderColor: 'border.emphasized' },
  '& [data-part="fill"]': {
    bg: 'bg.hover',
    borderRadius: `calc({radii.control} - ${1 + FILL_INSET_PX}px)`,
    insetBlock: `${FILL_INSET_PX}px`,
    insetInlineStart: `${FILL_INSET_PX}px`,
    pointerEvents: 'none',
    position: 'absolute',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'width',
  },
  '& [data-part="mark"]': {
    borderRadius: 'full',
    boxSize: '5px',
    pointerEvents: 'none',
    position: 'absolute',
    top: '50%',
    transform: 'translate(-50%, -50%)',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'opacity',
    '&[data-hidden]': { opacity: 0 },
    '&[data-state="under-value"]': { borderColor: 'border.emphasized', borderWidth: '1px' },
    '&[data-state="over-value"]': { bg: 'border.emphasized' },
    '&[data-state="at-value"]': { bg: 'accent.solid' },
  },
  '& [data-part="thumb"]': {
    bg: 'fg.muted',
    borderRadius: 'full',
    h: '3.5',
    pointerEvents: 'none',
    position: 'absolute',
    top: '50%',
    transform: 'translate(-50%, -50%)',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'left, background',
    w: '2px',
  },
  '&:hover [data-part="thumb"]': { bg: 'fg' },
  '&[data-dragging] [data-part="thumb"], &:has([data-part="slider"]:focus-visible) [data-part="thumb"]': {
    bg: 'accent.solid',
  },
  '&[data-dragging] [data-part="fill"], &[data-dragging] [data-part="thumb"]': { transitionProperty: 'none' },
  '& [data-part="slider"]': { inset: 0, outline: 'none', position: 'absolute' },
  '& [data-part="label"]': {
    ...fieldLabelRecipe.base,
    flexShrink: 1,
    minW: '0',
    overflow: 'hidden',
    ps: INSIDE_LABEL_INSET,
    position: 'relative',
    textOverflow: 'ellipsis',
    textShadow: SCRUBBER_TEXT_SHADOW,
    whiteSpace: 'nowrap',
  },
  '& [data-part="value"]': {
    alignSelf: 'stretch',
    appearance: 'none',
    bg: 'transparent',
    border: 0,
    color: 'fg',
    cursor: 'text',
    flexShrink: 0,
    font: 'inherit',
    fontVariantNumeric: 'tabular-nums',
    marginInlineStart: 'auto',
    minW: '0',
    outline: 'none',
    p: 0,
    pe: '3',
    position: 'relative',
    ps: '2',
    textAlign: 'end',
    textShadow: SCRUBBER_TEXT_SHADOW,
    whiteSpace: 'nowrap',
  },
  '& input[data-part="value"]': { flex: 1, userSelect: 'text', w: 'full' },
};

const readTextExtents = (root: HTMLElement): TextExtents | null => {
  const label = root.querySelector('[data-part="label"]');
  const value = root.querySelector('[data-part="value"]');

  if (!label || !value) {
    return null;
  }

  const rootRect = root.getBoundingClientRect();

  return {
    labelEnd: label.getBoundingClientRect().right - rootRect.left,
    valueStart: value.getBoundingClientRect().left - rootRect.left,
    width: rootRect.width,
  };
};

const isSameExtents = (a: TextExtents | null, b: TextExtents): boolean =>
  a !== null && a.labelEnd === b.labelEnd && a.valueStart === b.valueStart && a.width === b.width;

export const ScrubberField = ({
  defaultValue,
  disabled = false,
  error,
  formatValue,
  helpText,
  hint,
  inputMax,
  inputMin,
  label,
  marks,
  max,
  min,
  onChange,
  onChangeEnd,
  scale = 'linear',
  step,
  stepFor,
  value,
  ...stackProps
}: ScrubberFieldProps) => {
  const { t } = useTranslation();
  const id = useId();
  const labelId = `${id}-label`;
  const messageId = `${id}-message`;
  const sliderRef = useRef<HTMLDivElement>(null);
  const gestureRef = useRef<Gesture | null>(null);
  // `finished` guards the editor's blur, which fires while focus returns to the slider.
  const editSessionRef = useRef<{ finished: boolean } | null>(null);
  const [edit, setEdit] = useState<{ draft: string; selectAll: boolean } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [textExtents, setTextExtents] = useState<TextExtents | null>(null);

  useMountEffect(() => () => gestureRef.current?.settle());

  const measureText = useCallback((root: HTMLElement) => {
    const next = readTextExtents(root);

    if (next) {
      setTextExtents((current) => (isSameExtents(current, next) ? current : next));
    }
  }, []);
  const observeText = useCallback(
    (element: HTMLElement | null) => {
      const root = element?.closest<HTMLElement>('[data-scope="scrubber"]');

      if (!element || !root) {
        return undefined;
      }

      const observer = new ResizeObserver(() => measureText(root));

      observer.observe(element);

      return () => observer.disconnect();
    },
    [measureText]
  );

  // A log track needs a positive minimum; without one it falls back to linear rather than produce NaN.
  const isLog = scale === 'log' && min > 0;
  const span = isLog ? Math.log(max / min) : max - min;
  const decimals = Math.max(countDecimals(step), countDecimals(min));
  const typedMin = inputMin ?? min;
  const typedMax = inputMax ?? max;
  const snapToStep = useCallback(
    (raw: number, lower: number, upper: number): number =>
      clamp(roundTo(min + Math.round((raw - min) / step) * step, decimals), lower, upper),
    [decimals, min, step]
  );
  // Unclamped, so a drag anchored beyond either end still moves relative to the real value.
  const toTrack = useCallback(
    (target: number): number => {
      if (!(span > 0)) {
        return 0;
      }

      if (isLog) {
        return target > 0 ? Math.log(target / min) / span : Number.NEGATIVE_INFINITY;
      }

      return (target - min) / span;
    },
    [isLog, min, span]
  );
  const fromTrack = useCallback(
    (track: number): number => (isLog ? min * Math.exp(track * span) : min + track * span),
    [isLog, min, span]
  );
  const fractionOf = useCallback((target: number): number => clamp(toTrack(target), 0, 1), [toTrack]);
  const trackPosition = (fraction: number): string =>
    `calc(${TRACK_INSET_PX}px + ${fraction} * (100% - ${TRACK_INSET_PX * 2}px))`;
  const isUnderText = (fraction: number): boolean => {
    if (!textExtents) {
      return true;
    }

    const x = TRACK_INSET_PX + fraction * (textExtents.width - TRACK_INSET_PX * 2);

    return x < textExtents.labelEnd + TEXT_CLEARANCE_PX || x > textExtents.valueStart - TEXT_CLEARANCE_PX;
  };
  const valueFraction = fractionOf(value);
  const thumbPosition = trackPosition(valueFraction);
  const fillOverhang =
    FILL_OVERHANG_PX - FILL_INSET_PX + valueFraction * (TRACK_INSET_PX - FILL_INSET_PX - FILL_OVERHANG_PX);
  const formatted = formatValue ? formatValue(value) : String(value);
  const markEpsilon = step * MARK_EPSILON_RATIO;
  const markValues = useMemo(() => {
    const stops = defaultValue === undefined ? marks : [...(marks ?? []), defaultValue];

    return stops ? [...new Set(stops.filter((mark) => mark >= min && mark <= max))] : undefined;
  }, [defaultValue, marks, max, min]);
  const markEntries = useMemo(
    () =>
      markValues
        ?.filter((mark) => mark > min && mark < max)
        .map((mark) => {
          const state: ScrubberMarkState =
            Math.abs(mark - value) <= markEpsilon ? 'at-value' : mark < value ? 'over-value' : 'under-value';

          return { mark, state };
        }),
    [markEpsilon, markValues, max, min, value]
  );

  const commitValue = useCallback(
    (next: number) => {
      if (next === value) {
        return;
      }
      gestureRef.current?.settle();
      onChange(next);
      onChangeEnd?.(next);
    },
    [onChange, onChangeEnd, value]
  );

  const stepTo = useCallback(
    (next: number) => {
      if (next === value) {
        return;
      }

      const current = gestureRef.current;

      if (current?.kind === 'keys') {
        current.latest = next;
        onChange(next);
        return;
      }

      current?.settle();
      onChange(next);

      const keys: Gesture = {
        kind: 'keys',
        latest: next,
        settle: () => {
          if (gestureRef.current === keys) {
            gestureRef.current = null;
          }
          onChangeEnd?.(keys.latest);
        },
      };

      gestureRef.current = keys;
    },
    [onChange, onChangeEnd, value]
  );

  const startEditing = useCallback((draft: string, selectAll: boolean) => {
    editSessionRef.current = { finished: false };
    setEdit({ draft, selectAll });
  }, []);

  // Restore slider focus on Enter/Escape, never blur: Tab and clicks already chose a destination.
  const finishEditing = useCallback(
    (commit: boolean, restoreFocus: boolean) => {
      const session = editSessionRef.current;

      if (!session || session.finished) {
        return;
      }

      session.finished = true;
      setEdit(null);

      if (restoreFocus) {
        sliderRef.current?.focus({ preventScroll: true });
      }

      if (commit && edit) {
        const parsed = Number(edit.draft.trim());

        if (edit.draft.trim() !== '' && Number.isFinite(parsed)) {
          commitValue(clamp(parsed, typedMin, typedMax));
        }
      }
    },
    [commitValue, edit, typedMax, typedMin]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled || event.button !== 0 || edit || isValueTarget(event.target)) {
        return;
      }

      event.preventDefault();
      sliderRef.current?.focus({ preventScroll: true });
      gestureRef.current?.settle();

      const root = event.currentTarget;
      const rect = root.getBoundingClientRect();
      const trackLeft = rect.left + TRACK_INSET_PX;
      const trackWidth = Math.max(1, rect.width - TRACK_INSET_PX * 2);
      const trackAt = (clientX: number): number => (clientX - trackLeft) / trackWidth;
      const session = new AbortController();
      let isPendingTouch = event.pointerType === 'touch';
      let moved = false;
      // Re-anchor on sensitivity changes so toggling Shift mid-drag does not jump.
      let anchor = { clientX: event.clientX, ratio: event.shiftKey ? FINE_DRAG_RATIO : 1, track: toTrack(value) };
      // A value beyond the track stays reachable: dragging away holds it rather than snapping to the track's end.
      const lower = Math.min(min, value);
      const upper = Math.max(max, value);
      const drag: Gesture = {
        kind: 'drag',
        latest: value,
        settle: () => {
          session.abort();
          if (gestureRef.current === drag) {
            gestureRef.current = null;
          }
          setIsDragging(false);
          if (moved) {
            onChangeEnd?.(drag.latest);
          }
        },
      };

      gestureRef.current = drag;
      setIsDragging(true);

      // Pointer capture preserves drag cursor/hover isolation; window listeners cover capture failure.
      try {
        root.setPointerCapture(event.pointerId);
      } catch {
        // No capture available — fall through to the window listeners.
      }

      type PointerSample = { altKey: boolean; clientX: number; pointerId: number; shiftKey: boolean };
      const resolve = (pointer: PointerSample): number => {
        const ratio = pointer.shiftKey ? FINE_DRAG_RATIO : 1;

        if (anchor.ratio !== ratio) {
          anchor = { clientX: pointer.clientX, ratio, track: toTrack(drag.latest) };
        }

        const rawTrack = anchor.track + (trackAt(pointer.clientX) - trackAt(anchor.clientX)) * ratio;

        return pointer.altKey && markValues?.length
          ? nearestMark(rawTrack, markValues, toTrack)
          : snapToStep(fromTrack(rawTrack), lower, upper);
      };
      const apply = (pointer: PointerSample) => {
        if (pointer.pointerId !== event.pointerId) {
          return;
        }

        if (isPendingTouch) {
          if (Math.abs(pointer.clientX - event.clientX) < TOUCH_INTENT_PX) {
            return;
          }

          isPendingTouch = false;
        }

        const next = resolve(pointer);

        if (next !== drag.latest) {
          drag.latest = next;
          moved = true;
          onChange(next);
        }
      };
      const end = (pointer: PointerEvent) => {
        if (pointer.pointerId === event.pointerId) {
          drag.settle();
        }
      };

      window.addEventListener('pointermove', apply, { signal: session.signal });
      window.addEventListener('pointerup', end, { signal: session.signal });
      window.addEventListener('pointercancel', end, { signal: session.signal });
    },
    [disabled, edit, fromTrack, markValues, max, min, onChange, onChangeEnd, snapToStep, toTrack, value]
  );

  const handleDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!disabled && defaultValue !== undefined && !isValueTarget(event.target)) {
        commitValue(defaultValue);
      }
    },
    [commitValue, defaultValue, disabled]
  );

  // Consume handled keys so the window hotkey runtime cannot also treat editor-opening digits as commands.
  const handleSliderKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (disabled || event.ctrlKey || event.metaKey) {
        return;
      }

      let next: number | undefined;
      let precision = decimals;

      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowUp':
        case 'ArrowLeft':
        case 'ArrowDown':
        case 'PageUp':
        case 'PageDown': {
          const direction = event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'PageUp' ? 1 : -1;
          const coarse = event.shiftKey || event.key === 'PageUp' || event.key === 'PageDown';
          const fine = stepFor ? stepFor(value, direction) : step;
          next = value + direction * fine * (coarse ? COARSE_STEP_MULTIPLIER : 1);
          // A value-dependent step can be finer than `step`; rounding keeps its decimals.
          precision = Math.max(decimals, countDecimals(fine));
          break;
        }
        case 'Home':
          next = min;
          break;
        case 'End':
          next = max;
          break;
        case 'Enter':
        case 'F2':
          event.preventDefault();
          event.stopPropagation();
          startEditing(String(value), true);
          return;
        case 'Backspace':
        case 'Delete':
          // Owned even without a default: unconsumed, they reach delete hotkeys.
          event.preventDefault();
          event.stopPropagation();

          if (defaultValue !== undefined) {
            commitValue(defaultValue);
          }
          return;
        default:
          if (!event.altKey && /^[\d.-]$/.test(event.key)) {
            event.preventDefault();
            event.stopPropagation();
            startEditing(event.key, false);
          }
          return;
      }

      event.preventDefault();
      event.stopPropagation();
      stepTo(clamp(roundTo(next, precision), typedMin, typedMax));
    },
    [
      commitValue,
      decimals,
      defaultValue,
      disabled,
      max,
      min,
      startEditing,
      step,
      stepFor,
      stepTo,
      typedMax,
      typedMin,
      value,
    ]
  );

  const settleKeys = useCallback(() => {
    if (gestureRef.current?.kind === 'keys') {
      gestureRef.current.settle();
    }
  }, []);
  const handleSliderKeyUp = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (STEP_KEYS.has(event.key)) {
        settleKeys();
      }
    },
    [settleKeys]
  );

  const handleValueClick = useCallback(() => {
    if (!disabled) {
      startEditing(String(value), true);
    }
  }, [disabled, startEditing, value]);

  const handleInputChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const draft = event.target.value;

    setEdit((current) => (current ? { ...current, draft } : current));
  }, []);
  // Focus on attach: the editor mounts in response to the user's own click or keypress.
  const selectAll = edit?.selectAll ?? false;
  const attachEditor = useCallback(
    (input: HTMLInputElement | null) => {
      if (input) {
        input.focus();

        if (selectAll) {
          input.select();
        }
      }

      return observeText(input);
    },
    [observeText, selectAll]
  );
  const handleInputBlur = useCallback(
    (event: FocusEvent<HTMLInputElement>) => finishEditing(true, event.relatedTarget === null),
    [finishEditing]
  );
  const handleInputKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        finishEditing(true, true);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        finishEditing(false, true);
      }
    },
    [finishEditing]
  );

  const message = error ?? helpText;
  const labelContent = hint ? (
    <FeatureHint hint={hint}>
      <chakra.span>{label}</chakra.span>
    </FeatureHint>
  ) : (
    label
  );

  return (
    <Stack gap="1.5" minW="0" w="full" {...(hint ? FEATURE_HINT_FIELD_PROPS : undefined)} {...stackProps}>
      <Box
        ref={observeText}
        css={ROOT_CSS}
        data-disabled={disabled ? '' : undefined}
        data-dragging={isDragging ? '' : undefined}
        data-editing={edit ? '' : undefined}
        data-invalid={error ? '' : undefined}
        data-scope="scrubber"
        onDoubleClick={handleDoubleClick}
        onPointerDown={handlePointerDown}
      >
        <div data-part="fill" style={{ width: `calc(${thumbPosition} + ${fillOverhang}px)` }} />
        {markEntries?.map(({ mark, state }) => (
          <div
            key={mark}
            data-hidden={isUnderText(fractionOf(mark)) ? '' : undefined}
            data-part="mark"
            data-state={state}
            style={{ left: trackPosition(fractionOf(mark)) }}
          />
        ))}
        <div data-part="thumb" style={{ left: thumbPosition }} />
        <div
          ref={sliderRef}
          aria-describedby={message ? messageId : undefined}
          aria-disabled={disabled || undefined}
          aria-invalid={error ? true : undefined}
          aria-labelledby={labelId}
          aria-orientation="horizontal"
          aria-valuemax={Math.max(max, value)}
          aria-valuemin={Math.min(min, value)}
          aria-valuenow={value}
          aria-valuetext={formatted}
          data-part="slider"
          role="slider"
          tabIndex={disabled ? -1 : 0}
          onBlur={settleKeys}
          onKeyDown={handleSliderKeyDown}
          onKeyUp={handleSliderKeyUp}
        />
        <span ref={observeText} data-part="label" id={labelId}>
          {labelContent}
        </span>
        {edit ? (
          <input
            ref={attachEditor}
            aria-label={label}
            data-part="value"
            inputMode="decimal"
            type="text"
            value={edit.draft}
            onBlur={handleInputBlur}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
          />
        ) : (
          <button
            ref={observeText}
            aria-label={t('common.scrubber.editValue', { label, value: formatted })}
            data-part="value"
            disabled={disabled}
            tabIndex={-1}
            type="button"
            onClick={handleValueClick}
          >
            {formatted}
          </button>
        )}
      </Box>
      {message ? (
        <Text color={error ? 'fg.error' : 'fg.muted'} fontSize="xs" id={messageId} role={error ? 'alert' : undefined}>
          {message}
        </Text>
      ) : null}
    </Stack>
  );
};
