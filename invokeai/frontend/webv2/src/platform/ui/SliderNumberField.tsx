import type { SliderMark } from '@platform/ui/Slider';

import { HStack, NumberInput } from '@chakra-ui/react';
import { Slider } from '@platform/ui/Slider';
import { memo, useCallback, useMemo } from 'react';

type SliderNumberFieldProps = {
  ariaLabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  /** Drop off-track marks instead of clamping: clamping would falsely label a bound as the default. */
  marks?: SliderMark[];
  /** Looser clamps for typed values (slider bounds apply otherwise). */
  numberInputMin?: number;
  numberInputMax?: number;
  numberInputStep?: number;
  disabled?: boolean;
  formatValue?: (value: number) => string;
  onChange: (value: number) => void;
};

/**
 * Input bounds may exceed slider bounds. Callers own Field messaging and debouncing; parameter rows use
 * ScrubberField.
 */
export const SliderNumberField = memo(function SliderNumberField({
  ariaLabel,
  disabled,
  formatValue,
  marks,
  max,
  min,
  numberInputMax,
  numberInputMin,
  numberInputStep,
  onChange,
  step,
  value,
}: SliderNumberFieldProps) {
  const sliderAriaLabel = useMemo(() => [ariaLabel], [ariaLabel]);
  // Clamp only the thumb; preserve typed values within the input's wider bounds.
  const sliderValue = useMemo(() => [Math.min(max, Math.max(min, value))], [max, min, value]);
  const placeableMarks = marks?.filter((mark) => {
    const markValue = typeof mark === 'number' ? mark : mark.value;

    return markValue >= min && markValue <= max;
  });
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

  return (
    <HStack gap="2" w="full">
      <Slider
        aria-label={sliderAriaLabel}
        disabled={disabled}
        flex="1"
        formatValue={formatValue}
        marks={placeableMarks}
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
        <NumberInput.Input aria-label={ariaLabel} fontVariantNumeric="tabular-nums" />
      </NumberInput.Root>
    </HStack>
  );
});
