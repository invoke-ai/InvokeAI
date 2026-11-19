import type { FieldInputTemplate } from '@features/workflow/contracts';

import { Input } from '@chakra-ui/react';
import { Slider } from '@platform/ui/Slider';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FocusEvent,
  type MouseEvent,
  type WheelEvent,
} from 'react';

export const invalidProps = (invalid: boolean | undefined) => (invalid ? { 'aria-invalid': true } : {});

const toFiniteNumber = (raw: string): number | undefined => {
  if (raw.trim() === '') {
    return undefined;
  }

  const parsed = Number(raw);

  return Number.isFinite(parsed) ? parsed : undefined;
};

export const finiteNumberOrUndefined = (value: number | null | undefined): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const positiveFiniteNumberOrUndefined = (value: number | null | undefined): number | undefined => {
  const normalized = finiteNumberOrUndefined(value);

  return normalized !== undefined && normalized > 0 ? normalized : undefined;
};

/**
 * Held while focused: rendering the committed value lets its echo, which can lag a keystroke in Canvas, rewrite the
 * input under the caret.
 */
export const useFocusedDraft = () => {
  const [draft, setDraft] = useState<string | null>(null);
  const clearDraft = useCallback(() => setDraft(null), []);

  return [draft, setDraft, clearDraft] as const;
};

export type NumericInputTemplate = Pick<
  FieldInputTemplate,
  'exclusiveMaximum' | 'exclusiveMinimum' | 'maximum' | 'minimum' | 'multipleOf' | 'title' | 'type'
>;

export interface NumericInputProps {
  /** The accessible name when the visible label lives elsewhere; defaults to the template title. */
  ariaLabel?: string;
  disabled?: boolean;
  id?: string;
  invalid?: boolean;
  onChange: (value: number | undefined) => void;
  /** The arrow-key increment when the template declares no `multipleOf`. */
  step?: number;
  template: NumericInputTemplate;
  value: unknown;
}

// Tenths, as in v6; a template's `multipleOf` or the host's `step` takes precedence.
const FLOAT_STEP = '0.1';

const selectInputText = (event: MouseEvent<HTMLInputElement>) => event.currentTarget.select();

/** Every parseable entry commits unchanged; the host marks what cannot run instead of the control correcting it. */
export const NumericInput = ({
  ariaLabel,
  disabled,
  id,
  invalid,
  onChange,
  step,
  template,
  value,
}: NumericInputProps) => {
  const isInteger = template.type.name === 'IntegerField';
  const [draft, setDraft, clearDraft] = useFocusedDraft();
  // Set while a wheel event passes through: the blur it causes must not end the draft.
  const wheelRefocus = useRef(false);
  // Chromium keeps an unparseable partial entry (`-`, `1e`) on screen but reports it as empty, so the empty commit
  // alone cannot tell the field apart from a cleared one.
  const [hasBadInput, setHasBadInput] = useState(false);
  const text = draft ?? (typeof value === 'number' && Number.isFinite(value) ? String(value) : '');
  const min = finiteNumberOrUndefined(template.minimum) ?? finiteNumberOrUndefined(template.exclusiveMinimum);
  const max = finiteNumberOrUndefined(template.maximum) ?? finiteNumberOrUndefined(template.exclusiveMaximum);
  const multipleOf = positiveFiniteNumberOrUndefined(template.multipleOf);
  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const { validity, value: nextText } = event.currentTarget;

      setDraft(nextText);
      setHasBadInput(validity.badInput);
      // Committed as typed; the host marks fractional integers, out-of-range and empty required values instead.
      onChange(toFiniteNumber(nextText));
    },
    [onChange, setDraft]
  );
  const onBlur = useCallback(
    (event: FocusEvent<HTMLInputElement>) => {
      if (wheelRefocus.current) {
        return;
      }

      setHasBadInput(event.currentTarget.validity.badInput);
      clearDraft();
    },
    [clearDraft]
  );
  // Chromium steps a focused number input on wheel and swallows the scroll. Dropping focus for the event lets the panel
  // scroll instead; focus returns, with the draft and caret, once the scroll has been dispatched.
  const onWheel = useCallback(
    (event: WheelEvent<HTMLInputElement>) => {
      const input = event.currentTarget;

      if (document.activeElement !== input || wheelRefocus.current) {
        return;
      }

      wheelRefocus.current = true;
      input.blur();
      requestAnimationFrame(() => {
        wheelRefocus.current = false;

        if (input.isConnected && document.activeElement === document.body) {
          input.focus({ preventScroll: true });
        } else {
          // Focus went elsewhere within the frame (momentum wheel plus a click): finish the blur that was skipped.
          setHasBadInput(input.validity.badInput);
          clearDraft();
        }
      });
    },
    [clearDraft]
  );

  // Any non-empty text React writes replaces the bad entry on screen (a reset or undo landing a number), so the flag
  // cannot outlive it; while the entry is held the text is empty, so a lagging Canvas echo leaves it alone.
  if (hasBadInput && text !== '') {
    setHasBadInput(false);
  }

  return (
    <Input
      aria-label={ariaLabel ?? template.title}
      className="nodrag"
      disabled={disabled}
      id={id ? `${id}-number-input` : undefined}
      max={max !== undefined ? String(max) : undefined}
      min={min !== undefined ? String(min) : undefined}
      fontVariantNumeric="tabular-nums"
      step={
        multipleOf !== undefined ? String(multipleOf) : step !== undefined ? String(step) : isInteger ? '1' : FLOAT_STEP
      }
      type="number"
      value={text}
      w="full"
      {...invalidProps(invalid || hasBadInput)}
      onBlur={onBlur}
      onChange={onInputChange}
      onDoubleClick={selectInputText}
      onWheel={onWheel}
    />
  );
};

export interface NumericSliderInputProps {
  ariaLabel?: string;
  disabled?: boolean;
  range: { max: number; min: number; step: number };
  onChange: (value: number) => void;
  template: NumericInputTemplate;
  value: unknown;
  /** The combined control pairs the slider with a number input; the slider alone shows its value on the thumb. */
  withNumberInput: boolean;
}

/** An authored slider for a bounded number field; a value outside the track still moves the thumb to its end. */
export const NumericSliderInput = ({
  ariaLabel,
  disabled,
  onChange,
  range,
  template,
  value,
  withNumberInput,
}: NumericSliderInputProps) => {
  const label = ariaLabel ?? template.title;
  const current = typeof value === 'number' && Number.isFinite(value) ? value : range.min;
  const sliderAriaLabel = useMemo(() => [label], [label]);
  const sliderValue = useMemo(() => [Math.min(range.max, Math.max(range.min, current))], [current, range]);
  const onSliderChange = useCallback(
    ({ value: values }: { value: number[] }) => {
      const next = values[0];

      if (typeof next === 'number' && Number.isFinite(next)) {
        onChange(next);
      }
    },
    [onChange]
  );

  if (withNumberInput) {
    return (
      <SliderNumberField
        ariaLabel={label}
        disabled={disabled}
        max={range.max}
        min={range.min}
        step={range.step}
        value={current}
        onChange={onChange}
      />
    );
  }

  return (
    <Slider
      aria-label={sliderAriaLabel}
      className="nodrag"
      disabled={disabled}
      max={range.max}
      min={range.min}
      step={range.step}
      value={sliderValue}
      w="full"
      onValueChange={onSliderChange}
    />
  );
};
