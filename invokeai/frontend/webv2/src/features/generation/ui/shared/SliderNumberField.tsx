/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { SliderMark } from '@platform/ui/Slider';

import { HStack, InputGroup, NumberInput } from '@chakra-ui/react';
import { Slider } from '@platform/ui/Slider';

import { ModelDefaultButton } from './ModelDefaultButton';

interface SliderNumberFieldProps {
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
  /** Model default; the reset affordance shows only while the value differs. */
  defaultValue?: number;
  resetLabel?: string;
  disabled?: boolean;
  formatValue?: (value: number) => string;
  onChange: (value: number) => void;
}

/**
 * Slider + number input combo for numeric generation parameters. The slider
 * covers the practical range; the input accepts values beyond it when the
 * numberInput bounds are looser. Debouncing stays with the caller.
 */
export const SliderNumberField = ({
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
  step,
  value,
}: SliderNumberFieldProps) => {
  const showReset = defaultValue !== undefined && value !== defaultValue && !disabled;

  return (
    <HStack gap="2" w="full">
      <Slider
        aria-label={[ariaLabel]}
        disabled={disabled}
        flex="1"
        formatValue={formatValue}
        marks={marks}
        max={max}
        min={min}
        minW="0"
        size="sm"
        step={step}
        value={[value]}
        onValueChange={({ value: values }) => {
          const next = values[0];

          if (typeof next === 'number' && Number.isFinite(next)) {
            onChange(next);
          }
        }}
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
        onValueChange={({ valueAsNumber }) => {
          if (Number.isFinite(valueAsNumber)) {
            onChange(valueAsNumber);
          }
        }}
      >
        <InputGroup
          endElement={
            showReset ? (
              <ModelDefaultButton
                label={resetLabel}
                onClick={() => {
                  onChange(defaultValue);
                }}
              />
            ) : undefined
          }
          endElementProps={{ pointerEvents: 'auto' }}
        >
          <NumberInput.Input aria-label={ariaLabel} />
        </InputGroup>
      </NumberInput.Root>
    </HStack>
  );
};
