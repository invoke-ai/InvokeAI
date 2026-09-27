import type { FieldInputTemplate } from '@features/workflow/contracts';

import { Input } from '@chakra-ui/react';
import {
  useCallback,
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
 * The text exactly as typed, held while the control is focused and dropped on blur. Rendering the committed value
 * instead lets its echo rewrite the input under the caret (dropping a trailing `.` or `0`, moving the caret to the
 * end), and the Canvas rebuilds its flow model in a transition, so the echo can lag a keystroke behind.
 */
export const useFocusedDraft = () => {
  const [draft, setDraft] = useState<string | null>(null);
  const clearDraft = useCallback(() => setDraft(null), []);

  return [draft, setDraft, clearDraft] as const;
};

/** The slice of a field template the control reads: its kind, name, and the bounds it reports to the browser. */
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
  size?: 'xs' | '2xs';
  /** The arrow-key increment when the template declares no `multipleOf`. */
  step?: number;
  template: NumericInputTemplate;
  value: unknown;
}

/** A double-click anywhere in the box selects the whole value, not just the word under the pointer. */
const selectInputText = (event: MouseEvent<HTMLInputElement>) => event.currentTarget.select();

/**
 * The one numeric entry control for workflow values: the text stays as typed while focused, every parseable entry
 * commits unchanged, and the host marks what cannot run instead of the control correcting it.
 */
export const NumericInput = ({
  ariaLabel,
  disabled,
  id,
  invalid,
  onChange,
  size = 'xs',
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
      // Committed as typed: rounding, clamping, or dropping a sign would change what is on screen. The host marks
      // fractional integers, out-of-range values, and an empty required field invalid instead.
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
  // Chromium steps a focused number input on wheel and swallows the scroll. Dropping focus for the event lets the
  // panel scroll instead; focus returns, with the draft and caret, once the scroll has been dispatched.
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

  // Any non-empty text React writes replaces the bad entry on screen (a reset or undo landing a number), so the
  // flag cannot outlive it; while the entry is held the text is empty, so a lagging Canvas echo leaves it alone.
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
      size={size}
      step={multipleOf !== undefined ? String(multipleOf) : step !== undefined ? String(step) : isInteger ? '1' : 'any'}
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
