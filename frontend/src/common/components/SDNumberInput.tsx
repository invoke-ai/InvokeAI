import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Text,
  NumberInputProps,
} from '@chakra-ui/react';
import _ from 'lodash';
import { FocusEvent, useState } from 'react';

interface Props extends Omit<NumberInputProps, 'onChange'> {
  styleClass?: string;
  label?: string;
  width?: string | number;
  showStepper?: boolean;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}

/**
 * Customized Chakra FormControl + NumberInput multi-part component.
 */
const SDNumberInput = (props: Props) => {
  const {
    label,
    styleClass,
    isDisabled = false,
    showStepper = true,
    fontSize = '1rem',
    size = 'sm',
    width = '150px',
    textAlign,
    isInvalid,
    value,
    onChange,
    precision,
    min,
    max,
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

  const handleOnChange = (v: string) => {
    setValueAsString(v);

    /**
     * Cast the value to number.
     *
     * If there is no precision or the precision is 1, we infer that the value should be
     * an integer, and floor() it also.
     */
    if (Number(v) === _.clamp(Number(v), min, max)) {
      onChange(
        !precision || precision === 1 ? Math.floor(Number(v)) : Number(v)
      );
    }
  };

  /**
   * Clicking the steppers allows the value to go outside bounds; we need to
   * clamp it on blur and floor it if needed.
   */
  const handleBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      !precision || precision === 1
        ? Math.floor(Number(e.target.value))
        : Number(e.target.value),
      min,
      max
    );
    setValueAsString(String(clamped));
    onChange(clamped);
  };

  return (
    <FormControl
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className={`number-input ${styleClass}`}
    >
      {label && (
        <Text whiteSpace="nowrap" className="number-input-label">
          {label}
        </Text>
      )}
      <NumberInput
        size={size}
        {...rest}
        className="number-input-field"
        value={valueAsString}
        keepWithinRange={true}
        onChange={handleOnChange}
        onBlur={handleBlur}
      >
        <NumberInputField
          fontSize={fontSize}
          className="number-input-entry"
          width={width}
          textAlign={textAlign}
        />
        <div
          className="number-input-stepper"
          style={showStepper ? { display: 'block' } : { display: 'none' }}
        >
          <NumberIncrementStepper className="number-input-stepper-button" />
          <NumberDecrementStepper className="number-input-stepper-button" />
        </div>
      </NumberInput>
    </FormControl>
  );
};

export default SDNumberInput;
