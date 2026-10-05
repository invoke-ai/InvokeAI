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
import { FeatureHint } from '@platform/ui/hints';
import { fieldLabelRecipe, scrubberInteraction } from '@theme/recipes';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Track inset from the frame edge; the text padding clears it so the thumb never crosses a glyph. */
const TRACK_INSET_PX = 10;
/** The fill runs this far past the thumb so its rounded end frames it. */
const FILL_OVERHANG_PX = 6;
/** Clearance between the fill and the frame's border. */
const FILL_INSET_PX = 1.5;
/** Shift-drag moves the value this fraction of the pointer's track distance. */
const FINE_DRAG_RATIO = 0.1;
/** Shift/PageUp/PageDown step multiplier. */
const COARSE_STEP_MULTIPLIER = 10;
/** Marks land as `at-value` when within this fraction of a step. */
const MARK_EPSILON_RATIO = 1e-6;
/** Clearance between a stop and the label or value text before the stop hides. */
const TEXT_CLEARANCE_PX = 6;
// A panel-coloured halo keeps text legible where the fill passes behind it.
const SCRUBBER_TEXT_SHADOW = '0 1px 2px {colors.bg.panel}, 0 0 4px {colors.bg.panel}';
/** A touch must travel this far horizontally before it scrubs; until then the panel may pan. */
const TOUCH_INTENT_PX = 8;

type ScrubberMarkState = 'at-value' | 'over-value' | 'under-value';

/** Where the label ends and the value begins, in px from the frame's left edge. */
type TextExtents = { labelEnd: number; valueStart: number; width: number };

type OwnProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  /** Stops along the track; Alt-drag snaps to them. `defaultValue` is always a stop; stops at `min`/`max` are implied by the track ends and not drawn. */
  marks?: number[];
  /** Looser clamps for typed and keyboard values (the track bounds apply to scrubbing). */
  inputMin?: number;
  inputMax?: number;
  /** Double-click or Backspace/Delete restores this value. No reset without it. */
  defaultValue?: number;
  disabled?: boolean;
  hint?: FeatureHintId;
  /** Validation error, shown in place of `helpText`; also marks the field invalid. */
  error?: string | null;
  helpText?: string;
  /** Formats the displayed value and its `aria-valuetext`; typing always edits the raw number. */
  formatValue?: (value: number) => string;
  /** Fires per step of a gesture. A drag keeps the handler it started with, so patch-style handlers only. */
  onChange: (value: number) => void;
};

export type ScrubberFieldProps = OwnProps & Omit<StackProps, keyof OwnProps | 'children' | 'defaultValue' | 'onChange'>;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const countDecimals = (value: number): number => {
  const text = String(value);
  const dot = text.indexOf('.');

  return dot === -1 ? 0 : text.length - dot - 1;
};

const roundTo = (value: number, decimals: number): number => Number(value.toFixed(decimals));

const nearestMark = (raw: number, marks: readonly number[]): number =>
  marks.reduce((best, mark) => (Math.abs(mark - raw) < Math.abs(best - raw) ? mark : best));

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
    // The same pointed tint as list rows, so a field and the row it sits in share one interaction color.
    bg: 'bg.hover',
    // Concentric with the frame's corner inside its 1px border and the fill's clearance.
    borderRadius: `calc({radii.control} - ${1 + FILL_INSET_PX}px)`,
    insetBlock: `${FILL_INSET_PX}px`,
    insetInlineStart: `${FILL_INSET_PX}px`,
    pointerEvents: 'none',
    position: 'absolute',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'width',
  },
  // Stops share the centerline with the text; one under a glyph hides rather than colliding.
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
    ps: '3',
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

/**
 * Relative drag; Shift fine, Alt stops. Click the value to edit; arrows step, Shift/Page keys ×10, Home/End
 * bounds. Double-click or Backspace/Delete resets. Caller owns debouncing.
 */
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
  step,
  value,
  ...stackProps
}: ScrubberFieldProps) => {
  const { t } = useTranslation();
  const id = useId();
  const labelId = `${id}-label`;
  const messageId = `${id}-message`;
  const sliderRef = useRef<HTMLDivElement>(null);
  const pointerSessionRef = useRef<AbortController | null>(null);
  // `finished` guards the editor's blur, which fires while focus returns to the slider.
  const editSessionRef = useRef<{ finished: boolean } | null>(null);
  const [edit, setEdit] = useState<{ draft: string; selectAll: boolean } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [textExtents, setTextExtents] = useState<TextExtents | null>(null);

  useMountEffect(() => () => pointerSessionRef.current?.abort());

  // Track label/value bounds so stops stay hidden beneath changing text.
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

  const range = max - min;
  const decimals = Math.max(countDecimals(step), countDecimals(min));
  const typedMin = inputMin ?? min;
  const typedMax = inputMax ?? max;
  const snapToStep = useCallback(
    (raw: number): number => clamp(roundTo(min + Math.round((raw - min) / step) * step, decimals), min, max),
    [decimals, max, min, step]
  );
  const fractionOf = useCallback(
    (target: number): number => (range > 0 ? clamp((target - min) / range, 0, 1) : 0),
    [min, range]
  );
  const trackPosition = (fraction: number): string =>
    `calc(${TRACK_INSET_PX}px + ${fraction} * (100% - ${TRACK_INSET_PX * 2}px))`;
  const isUnderText = (fraction: number): boolean => {
    if (!textExtents) {
      return true;
    }

    const x = TRACK_INSET_PX + fraction * (textExtents.width - TRACK_INSET_PX * 2);

    return x < textExtents.labelEnd + TEXT_CLEARANCE_PX || x > textExtents.valueStart - TEXT_CLEARANCE_PX;
  };
  const thumbPosition = trackPosition(fractionOf(value));
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

  const emit = useCallback(
    (next: number) => {
      if (next !== value) {
        onChange(next);
      }
    },
    [onChange, value]
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
          emit(clamp(parsed, typedMin, typedMax));
        }
      }
    },
    [edit, emit, typedMax, typedMin]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled || event.button !== 0 || edit || isValueTarget(event.target)) {
        return;
      }

      event.preventDefault();
      sliderRef.current?.focus({ preventScroll: true });
      pointerSessionRef.current?.abort();

      const root = event.currentTarget;
      const rect = root.getBoundingClientRect();
      const trackLeft = rect.left + TRACK_INSET_PX;
      const trackWidth = Math.max(1, rect.width - TRACK_INSET_PX * 2);
      // Unclamped, so a press left of the thumb can still be dragged past the track's end to max.
      const valueAt = (clientX: number): number => min + ((clientX - trackLeft) / trackWidth) * range;
      const session = new AbortController();
      let latest = value;
      let isPendingTouch = event.pointerType === 'touch';
      // Re-anchor on sensitivity changes so toggling Shift mid-drag does not jump.
      let anchor = { clientX: event.clientX, ratio: event.shiftKey ? FINE_DRAG_RATIO : 1, value };

      pointerSessionRef.current = session;
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
          anchor = { clientX: pointer.clientX, ratio, value: latest };
        }

        const raw = anchor.value + (valueAt(pointer.clientX) - valueAt(anchor.clientX)) * ratio;

        return pointer.altKey && markValues?.length ? nearestMark(raw, markValues) : snapToStep(raw);
      };
      // Only the initiating pointer may move or end this gesture.
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

        if (next !== latest) {
          latest = next;
          onChange(next);
        }
      };
      const end = (pointer: PointerEvent) => {
        if (pointer.pointerId !== event.pointerId) {
          return;
        }

        session.abort();
        pointerSessionRef.current = null;
        setIsDragging(false);
      };

      window.addEventListener('pointermove', apply, { signal: session.signal });
      window.addEventListener('pointerup', end, { signal: session.signal });
      window.addEventListener('pointercancel', end, { signal: session.signal });
    },
    [disabled, edit, markValues, min, onChange, range, snapToStep, value]
  );

  const handleDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!disabled && defaultValue !== undefined && !isValueTarget(event.target)) {
        emit(defaultValue);
      }
    },
    [defaultValue, disabled, emit]
  );

  // Consume handled keys so the window hotkey runtime cannot also treat editor-opening digits as commands.
  const handleSliderKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (disabled || event.ctrlKey || event.metaKey) {
        return;
      }

      const coarse = step * COARSE_STEP_MULTIPLIER;
      const stepBy = event.shiftKey ? coarse : step;
      let next: number | undefined;

      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowUp':
          next = value + stepBy;
          break;
        case 'ArrowLeft':
        case 'ArrowDown':
          next = value - stepBy;
          break;
        case 'PageUp':
          next = value + coarse;
          break;
        case 'PageDown':
          next = value - coarse;
          break;
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
            emit(defaultValue);
          }
          return;
        default:
          // A digit (or sign/point) starts typing straight away, like a spreadsheet cell.
          if (!event.altKey && /^[\d.-]$/.test(event.key)) {
            event.preventDefault();
            event.stopPropagation();
            startEditing(event.key, false);
          }
          return;
      }

      event.preventDefault();
      event.stopPropagation();
      emit(clamp(roundTo(next, decimals), typedMin, typedMax));
    },
    [decimals, defaultValue, disabled, emit, max, min, startEditing, step, typedMax, typedMin, value]
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
  // With no relatedTarget, retain the slider tab stop.
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
    <Stack gap="1.5" minW="0" w="full" {...stackProps}>
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
        <div data-part="fill" style={{ width: `calc(${thumbPosition} + ${FILL_OVERHANG_PX - FILL_INSET_PX}px)` }} />
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
          onKeyDown={handleSliderKeyDown}
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
            aria-label={t('common.scrubber.editValue', { label })}
            data-part="value"
            disabled={disabled}
            // One tab stop per field: the slider opens the editor with Enter or a digit.
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
