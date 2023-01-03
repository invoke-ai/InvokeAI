import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberIncrementStepper,
  NumberDecrementStepper,
  NumberInputProps,
  FormLabel,
  NumberInputFieldProps,
  NumberInputStepperProps,
  FormControlProps,
  FormLabelProps,
  TooltipProps,
  Tooltip,
} from '@chakra-ui/react';
import _ from 'lodash';
import { FocusEvent, useEffect, useState } from 'react';

const numberStringRegex = /^-?(0\.)?\.?$/;

interface Props extends Omit<NumberInputProps, 'onChange'> {
  styleClass?: string;
  label?: string;
  labelFontSize?: string | number;
  width?: string | number;
  showStepper?: boolean;
  value?: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  clamp?: boolean;
  isInteger?: boolean;
  formControlProps?: FormControlProps;
  formLabelProps?: FormLabelProps;
  numberInputProps?: NumberInputProps;
  numberInputFieldProps?: NumberInputFieldProps;
  numberInputStepperProps?: NumberInputStepperProps;
  tooltipProps?: Omit<TooltipProps, 'children'>;
}

/**
 * Customized Chakra FormControl + NumberInput multi-part component.
 */
const IAINumberInput = (props: Props) => {
  const {
    label,
    labelFontSize = 'sm',
    styleClass,
    isDisabled = false,
    showStepper = true,
    width,
    textAlign,
    isInvalid,
    value,
    onChange,
    min,
    max,
    isInteger = true,
    formControlProps,
    formLabelProps,
    numberInputFieldProps,
    numberInputStepperProps,
    tooltipProps,
    ...rest
  } = props;

  /**
   * Using a controlled input with a value that accepts decimals needs special
   * handling. If the user starts to type in "1.5", by the time they press the
   * 5, the value has been parsed from "1." to "1" and they end up with "15".
   *
   * To resolve this, this component keeps a the value as a string internally,
   * and the UI component uses that. When a change is made, that string is parsed
   * as a number and given to the `onChange` function.
   */

  const [valueAsString, setValueAsString] = useState<string>(String(value));

  /**
   * When `value` changes (e.g. from a diff source than this component), we need
   * to update the internal `valueAsString`, but only if the actual value is different
   * from the current value.
   */
  useEffect(() => {
    if (
      !valueAsString.match(numberStringRegex) &&
      value !== Number(valueAsString)
    ) {
      setValueAsString(String(value));
    }
  }, [value, valueAsString]);

  const handleOnChange = (v: string) => {
    setValueAsString(v);
    // This allows negatives and decimals e.g. '-123', `.5`, `-0.2`, etc.
    if (!v.match(numberStringRegex)) {
      // Cast the value to number. Floor it if it should be an integer.
      onChange(isInteger ? Math.floor(Number(v)) : Number(v));
    }
  };

  /**
   * Clicking the steppers allows the value to go outside bounds; we need to
   * clamp it on blur and floor it if needed.
   */
  const handleBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(e.target.value),
      min,
      max
    );
    setValueAsString(String(clamped));
    onChange(clamped);
  };

  return (
    <Tooltip {...tooltipProps}>
      <FormControl
        isDisabled={isDisabled}
        isInvalid={isInvalid}
        className={
          styleClass
            ? `invokeai__number-input-form-control ${styleClass}`
            : `invokeai__number-input-form-control`
        }
        {...formControlProps}
      >
        {label && (
          <FormLabel
            className="invokeai__number-input-form-label"
            style={{ display: label ? 'block' : 'none' }}
            fontSize={labelFontSize}
            fontWeight="bold"
            marginRight={0}
            marginBottom={0}
            whiteSpace="nowrap"
            {...formLabelProps}
          >
            {label}
          </FormLabel>
        )}
        <NumberInput
          className="invokeai__number-input-root"
          value={valueAsString}
          min={min}
          max={max}
          keepWithinRange={true}
          clampValueOnBlur={false}
          onChange={handleOnChange}
          onBlur={handleBlur}
          width={width}
          {...rest}
        >
          <NumberInputField
            className="invokeai__number-input-field"
            textAlign={textAlign}
            {...numberInputFieldProps}
          />
          {showStepper && (
            <div className="invokeai__number-input-stepper">
              <NumberIncrementStepper
                {...numberInputStepperProps}
                className="invokeai__number-input-stepper-button"
              />
              <NumberDecrementStepper
                {...numberInputStepperProps}
                className="invokeai__number-input-stepper-button"
              />
            </div>
          )}
        </NumberInput>
      </FormControl>
    </Tooltip>
  );
};

export default IAINumberInput;
