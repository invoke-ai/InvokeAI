/* oxlint-disable react-perf/jsx-no-jsx-as-prop */
import type { NumberInput as ChakraNumberInput } from '@chakra-ui/react';
import type { InputEvent, KeyboardEvent, ReactNode } from 'react';

import { Box, Flex, Icon, InputGroup, NumberInput, VisuallyHidden } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { CONTROL_HEIGHT_PX } from '@theme/scale';
import { MoveHorizontalIcon } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { parseRelativeInput, resolveRelativeValue, type RelativeOperator } from './relativeValue';

/** Width of every numeric field, so X / Y / W / H line up across tools; fits "-1234.56" beside its affixes. */
const FORM_NUMBER_FIELD_WIDTH_PX = 80;

interface FormNumberFieldProps {
  'aria-label': string;
  disabled?: boolean;
  /** Short prefix drawn inside the field (X, Y, W, H); it doubles as the scrub handle. */
  label?: string;
  max?: number;
  min?: number;
  /** Commit mode only: also accept `N%`, `*N` and `/N` of the current value; `min` then bounds the result. */
  relative?: boolean;
  step?: number;
  suffix?: string;
  value: string;
  /** Live mode: every keystroke reaches the owner (session previews). */
  onValueChange?: (details: ChakraNumberInput.ValueChangeDetails) => void;
  /** Commit mode: the field keeps a draft and hands it over on blur, Enter or the end of a scrub (document edits). */
  onValueCommit?: (details: ChakraNumberInput.ValueChangeDetails) => void;
}

const AFFIX_PROPS = { color: 'fg.muted', fontSize: 'xs', lineHeight: '1' } as const;
// InputGroup pads the input by `--input-height` minus the offset; the text must clear the affix.
const INPUT_HEIGHT_PX = CONTROL_HEIGHT_PX.md;
const AFFIX_CHAR_PX = 6;
const AFFIX_GUTTER_PX = 8;
const affixOffset = (chars: number): string => `${INPUT_HEIGHT_PX - AFFIX_GUTTER_PX - chars * AFFIX_CHAR_PX}px`;
const START_ELEMENT_PROPS = { ...AFFIX_PROPS, fontWeight: 'medium', pointerEvents: 'auto', ps: '1.5' } as const;
const END_ELEMENT_PROPS = { ...AFFIX_PROPS, pe: '1.5', pointerEvents: 'none' } as const;
const OPERATOR_PREFIX: Partial<Record<RelativeOperator, string>> = { '*': '×', '/': '÷' };
const OPERATOR_DESCRIPTION_KEYS: Record<RelativeOperator, string> = {
  '%': 'widgets.canvas.toolOptions.relativePercent',
  '*': 'widgets.canvas.toolOptions.relativeMultiply',
  '/': 'widgets.canvas.toolOptions.relativeDivide',
};

/** Drafts commit once on blur, Enter or release, and only after edits, so tabbing keeps precise values. */
export const FormNumberField = ({
  'aria-label': ariaLabel,
  disabled,
  label,
  max,
  min,
  relative = false,
  step,
  suffix,
  value,
  onValueChange,
  onValueCommit,
}: FormNumberFieldProps) => {
  const { t } = useTranslation();
  const operatorDescriptionId = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const draftRef = useRef<string | null>(null);
  const [operator, setOperator] = useState<RelativeOperator | null>(null);
  const operatorRef = useRef<RelativeOperator | null>(null);
  const scrubEnd = useRef<(() => void) | null>(null);
  const setDraftText = useCallback((next: string | null) => {
    draftRef.current = next;
    setDraft(next);
  }, []);
  const setOperatorState = useCallback((next: RelativeOperator | null) => {
    operatorRef.current = next;
    setOperator(next);
  }, []);
  const onDraftChange = useCallback(
    ({ value: next }: ChakraNumberInput.ValueChangeDetails) => setDraftText(next),
    [setDraftText]
  );
  const onDraftCommit = useCallback(
    (details: ChakraNumberInput.ValueChangeDetails) => {
      const edited = draftRef.current !== null;
      const pending = operatorRef.current;
      setDraftText(null);
      setOperatorState(null);
      if (pending) {
        const result = resolveRelativeValue(Number.parseFloat(value), pending, details.valueAsNumber);
        if (result !== null && result > 0 && String(result) !== value) {
          onValueCommit?.({ value: String(result), valueAsNumber: result });
        }
        return;
      }
      if (edited && details.value !== value) {
        onValueCommit?.(details);
      }
    },
    [onValueCommit, setDraftText, setOperatorState, value]
  );
  // Caught before the number input, which rejects any character that is not part of a number.
  const onBeforeInputCapture = useCallback(
    (event: InputEvent<HTMLDivElement>) => {
      const input = event.target as HTMLInputElement;
      const parsed = parseRelativeInput(event.data);
      if (!parsed) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      setOperatorState(parsed.operator);
      setDraftText(parsed.operand || (parsed.operator === '%' ? input.value : ''));
    },
    [setDraftText, setOperatorState]
  );
  const onKeyDownCapture = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (!operatorRef.current) {
        return;
      }
      const input = event.target as HTMLInputElement;
      if (event.key === 'Escape' || (event.key === 'Backspace' && input.value === '')) {
        event.preventDefault();
        setOperatorState(null);
        if (event.key === 'Escape') {
          setDraftText(null);
        }
      }
    },
    [setDraftText, setOperatorState]
  );
  // The number input commits on blur and Enter only; a scrub ends with the mouse button, so the draft it produced is
  // committed when the button lifts. The listener reads the ref: a release in the same frame as the last move must not
  // commit a draft one step behind.
  const onScrubStart = useCallback(() => {
    scrubEnd.current?.();
    if (operatorRef.current) {
      setOperatorState(null);
      setDraftText(null);
    }
    const scrub = new AbortController();
    scrubEnd.current = () => scrub.abort();
    window.addEventListener(
      'mouseup',
      () => {
        scrub.abort();
        scrubEnd.current = null;
        const next = draftRef.current;
        if (next !== null) {
          onDraftCommit({ value: next, valueAsNumber: Number.parseFloat(next) });
        }
      },
      { signal: scrub.signal }
    );
  }, [onDraftCommit, setDraftText, setOperatorState]);
  useEffect(() => () => scrubEnd.current?.(), []);
  const live = onValueChange !== undefined;
  const prefix = operator ? OPERATOR_PREFIX[operator] : undefined;
  const shownSuffix = operator === '%' ? '%' : suffix;
  const startChars = (label?.length ?? 2) + (prefix ? 2 : 0);
  return (
    <NumberInput.Root
      disabled={disabled}
      flexShrink={0}
      max={max}
      // The bound applies to the result, so an operand such as `*0.5` is not clamped on blur.
      min={operator ? undefined : min}
      step={step}
      value={live ? value : (draft ?? value)}
      w={`${FORM_NUMBER_FIELD_WIDTH_PX}px`}
      onBeforeInputCapture={relative && !live ? onBeforeInputCapture : undefined}
      onKeyDownCapture={relative && !live ? onKeyDownCapture : undefined}
      onValueChange={live ? onValueChange : onDraftChange}
      onValueCommit={live ? undefined : onDraftCommit}
    >
      <InputGroup
        endElement={shownSuffix ? <span aria-hidden>{shownSuffix}</span> : undefined}
        endElementProps={END_ELEMENT_PROPS}
        endOffset={shownSuffix ? affixOffset(shownSuffix.length) : undefined}
        startElement={
          <>
            <NumberInput.Scrubber aria-hidden onMouseDownCapture={live ? undefined : onScrubStart}>
              {label ?? <Icon as={MoveHorizontalIcon} boxSize="3" />}
            </NumberInput.Scrubber>
            {prefix ? (
              <Box aria-hidden as="span" color="fg" ms="1">
                {prefix}
              </Box>
            ) : null}
          </>
        }
        startElementProps={START_ELEMENT_PROPS}
        startOffset={affixOffset(startChars)}
      >
        <NumberInput.Input
          aria-describedby={operator ? operatorDescriptionId : undefined}
          aria-label={ariaLabel}
          fontVariantNumeric="tabular-nums"
          textAlign="end"
        />
      </InputGroup>
      {operator ? (
        <VisuallyHidden id={operatorDescriptionId}>{t(OPERATOR_DESCRIPTION_KEYS[operator])}</VisuallyHidden>
      ) : null}
    </NumberInput.Root>
  );
};

/** Commit-mode handler: applies a finite number and ignores the rest. */
export const useNumberCommit = (apply: (value: number) => void) =>
  useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        apply(valueAsNumber);
      }
    },
    [apply]
  );

/** Slider gesture state: `settle` runs once on release (one history entry); `preview` mirrors ticks. */
export const useSliderGesture = (
  committed: number,
  settle: (value: number) => void,
  preview?: (value: number) => void
) => {
  const [draft, setDraft] = useState<number | null>(null);
  // The end event also fires for gestures that changed nothing (a thumb click, an arrow key at the boundary); settling
  // those would record empty history entries, so a gesture only settles a genuinely different value.
  const committedRef = useRef(committed);
  useEffect(() => {
    committedRef.current = committed;
  }, [committed]);
  const onChange = useCallback(
    (value: number) => {
      setDraft(value);
      preview?.(value);
    },
    [preview]
  );
  const onChangeEnd = useCallback(
    (value: number) => {
      setDraft(null);
      if (value !== committedRef.current) {
        settle(value);
      }
    },
    [settle]
  );
  return { onChange, onChangeEnd, value: draft ?? committed };
};

/** Apply and Cancel for a pending operation, session or float, after whatever the owner shows about it. */
export const ApplyCancelBar = ({
  applyDisabled = true,
  applyLoading = false,
  applyLabel,
  cancelDisabled = true,
  children,
  onApply,
  onCancel,
}: {
  applyDisabled?: boolean;
  applyLoading?: boolean;
  applyLabel?: ReactNode;
  cancelDisabled?: boolean;
  children?: ReactNode;
  onApply?: () => void;
  onCancel?: () => void;
}) => {
  const { t } = useTranslation();
  return (
    <Flex align="center" gap="1" minW="0" w="full">
      <Box flex="1" minW="0" overflow="hidden" whiteSpace="nowrap">
        {children}
      </Box>
      <Button
        data-pane-action="apply"
        disabled={applyDisabled}
        flexShrink={0}
        loading={applyLoading}
        variant="solid"
        onClick={onApply}
      >
        {applyLabel ?? t('common.apply')}
      </Button>
      <Button data-pane-action="cancel" disabled={cancelDisabled} flexShrink={0} variant="ghost" onClick={onCancel}>
        {t('common.cancel')}
      </Button>
    </Flex>
  );
};
