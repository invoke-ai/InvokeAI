import { forwardRef, NumberInput as ChakraNumberInput } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { $shift } from 'common/hooks/useGlobalModifiers';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import { clamp } from 'lodash-es';
import type { FocusEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';

import { InvNumberInputField } from './InvNumberInputField';
import { InvNumberInputStepper } from './InvNumberInputStepper';
import type { InvNumberInputProps } from './types';

const isValidCharacter = (char: string) => /^[0-9\-.]$/i.test(char);

export const InvNumberInput = memo(
  forwardRef<InvNumberInputProps, typeof ChakraNumberInput>(
    (props: InvNumberInputProps, ref) => {
      const {
        value,
        min = 0,
        max,
        step: _step = 1,
        fineStep: _fineStep,
        onChange: _onChange,
        numberInputFieldProps,
        defaultValue,
        ...rest
      } = props;

      const [valueAsString, setValueAsString] = useState<string>(String(value));
      const [valueAsNumber, setValueAsNumber] = useState<number>(value);
      const shift = useStore($shift);
      const step = useMemo(
        () => (shift ? _fineStep ?? _step : _step),
        [shift, _fineStep, _step]
      );
      const isInteger = useMemo(
        () => Number.isInteger(_step) && Number.isInteger(_fineStep ?? 1),
        [_step, _fineStep]
      );

      const inputMode = useMemo(
        () => (isInteger ? 'numeric' : 'decimal'),
        [isInteger]
      );

      const precision = useMemo(() => (isInteger ? 0 : 3), [isInteger]);

      const onChange = useCallback(
        (valueAsString: string, valueAsNumber: number) => {
          setValueAsString(valueAsString);
          if (isNaN(valueAsNumber)) {
            return;
          }
          setValueAsNumber(valueAsNumber);
          _onChange(valueAsNumber);
        },
        [_onChange]
      );

      // This appears to be unnecessary? Cannot figure out what it did but leaving it here in case
      // it was important.
      // const onClickStepper = useCallback(
      //   () => _onChange(Number(valueAsString)),
      //   [_onChange, valueAsString]
      // );

      const onBlur: FocusEventHandler<HTMLInputElement> = useCallback(
        (e) => {
          if (!e.target.value) {
            // If the input is empty, we set it to the minimum value
            onChange(String(defaultValue ?? min), Number(defaultValue) ?? min);
          } else {
            // Otherwise, we round the value to the nearest multiple if integer, else 3 decimals
            const roundedValue = isInteger
              ? roundToMultiple(valueAsNumber, _fineStep ?? _step)
              : Number(valueAsNumber.toFixed(precision));
            // Clamp to min/max
            const clampedValue = clamp(roundedValue, min, max);
            onChange(String(clampedValue), clampedValue);
          }
        },
        [
          _fineStep,
          _step,
          defaultValue,
          isInteger,
          max,
          min,
          onChange,
          precision,
          valueAsNumber,
        ]
      );

      /**
       * When `value` changes (e.g. from a diff source than this component), we need
       * to update the internal `valueAsString`, but only if the actual value is different
       * from the current value.
       */
      useEffect(() => {
        if (value !== valueAsNumber) {
          setValueAsString(String(value));
          setValueAsNumber(value);
        }
      }, [value, valueAsNumber]);

      return (
        <ChakraNumberInput
          ref={ref}
          value={valueAsString}
          defaultValue={defaultValue}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          clampValueOnBlur={false}
          isValidCharacter={isValidCharacter}
          focusInputOnChange={false}
          onPaste={stopPastePropagation}
          inputMode={inputMode}
          precision={precision}
          variant="filled"
          {...rest}
        >
          <InvNumberInputField onBlur={onBlur} {...numberInputFieldProps} />
          <InvNumberInputStepper />
        </ChakraNumberInput>
      );
    }
  )
);

InvNumberInput.displayName = 'InvNumberInput';
