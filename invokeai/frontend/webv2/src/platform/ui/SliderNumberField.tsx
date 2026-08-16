import type { SliderMark } from '@platform/ui/Slider';

import { HStack, InputGroup, NumberInput } from '@chakra-ui/react';
import { MODEL_DEFAULT_END_ELEMENT_PROPS, ModelDefaultButton } from '@platform/ui/ModelDefaultButton';
import { Slider } from '@platform/ui/Slider';
import { memo, useCallback, useMemo } from 'react';

/**
 * `resetLabel` is required exactly when `defaultValue` is wired up — there's no
 * dead fallback to fall back to. `defaultValue` itself may still resolve to
 * `undefined` at runtime (e.g. model defaults not loaded yet); the reset
 * affordance simply stays hidden until it does.
 */
type SliderNumberFieldResetProps =
  | { defaultValue?: undefined; resetLabel?: undefined }
  | { defaultValue: number | undefined; resetLabel: string };

type SliderNumberFieldProps = {
  ariaLabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  marks?: SliderMark[];
  /** Looser clamps for typed values (slider bounds apply otherwise). */
  numberInputMin?: number;
  numberInputMax?: number;
  numberInputStep?: number;
  /**
   * Renders the increment/decrement stepper. Off by default. Not meant to be
   * combined with `defaultValue`/`resetLabel` — the stepper and the reset
   * button's `InputGroup` endElement compete for the same trailing slot; no
   * current caller does both, and nothing enforces it, so don't be the first.
   */
  showStepper?: boolean;
  disabled?: boolean;
  formatValue?: (value: number) => string;
  onChange: (value: number) => void;
} & SliderNumberFieldResetProps;

/**
 * Slider + number input combo for numeric parameters. The slider covers the
 * practical range; the input accepts values beyond it when the numberInput
 * bounds are looser. Debouncing stays with the caller. Label, hint, and
 * validation messaging are the caller's job (compose with `Field`) — this
 * component only owns the slider/input pairing.
 */
export const SliderNumberField = memo(function SliderNumberField({
  ariaLabel,
  defaultValue,
  disabled,
  formatValue,
  marks,
  max,
  min,
  numberInputMax,
  numberInputMin,
  numberInputStep,
  onChange,
  resetLabel,
  showStepper = false,
  step,
  value,
}: SliderNumberFieldProps) {
  const sliderAriaLabel = useMemo(() => [ariaLabel], [ariaLabel]);
  // Typed values may exceed the slider's own range (the number input has its own,
  // looser bounds via numberInputMin/Max); the thumb clamps to stay on the track
  // instead of rendering off it, while the input keeps showing the typed value.
  const sliderValue = useMemo(() => [Math.min(max, Math.max(min, value))], [max, min, value]);
  const handleSliderChange = useCallback(
    ({ value: values }: { value: number[] }) => {
      const next = values[0];

      if (typeof next === 'number' && Number.isFinite(next)) {
        onChange(next);
      }
    },
    [onChange]
  );
  const handleNumberChange = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onChange(valueAsNumber);
      }
    },
    [onChange]
  );
  const handleReset = useCallback(() => {
    if (defaultValue !== undefined) {
      onChange(defaultValue);
    }
  }, [defaultValue, onChange]);
  const resetElement = useMemo(() => {
    if (defaultValue === undefined || resetLabel === undefined || disabled || value === defaultValue) {
      return undefined;
    }

    return <ModelDefaultButton label={resetLabel} onClick={handleReset} />;
  }, [defaultValue, disabled, handleReset, resetLabel, value]);

  return (
    <HStack gap="2" w="full">
      <Slider
        aria-label={sliderAriaLabel}
        disabled={disabled}
        flex="1"
        formatValue={formatValue}
        marks={marks}
        max={max}
        min={min}
        minW="0"
        size="sm"
        step={step}
        value={sliderValue}
        onValueChange={handleSliderChange}
      />
      <NumberInput.Root
        disabled={disabled}
        flexShrink="0"
        max={numberInputMax ?? max}
        min={numberInputMin ?? min}
        size="xs"
        step={numberInputStep ?? step}
        value={String(value)}
        w="20"
        onValueChange={handleNumberChange}
      >
        {showStepper ? <NumberInput.Control /> : null}
        <InputGroup endElement={resetElement} endElementProps={MODEL_DEFAULT_END_ELEMENT_PROPS}>
          <NumberInput.Input aria-label={ariaLabel} fontVariantNumeric="tabular-nums" />
        </InputGroup>
      </NumberInput.Root>
    </HStack>
  );
});
