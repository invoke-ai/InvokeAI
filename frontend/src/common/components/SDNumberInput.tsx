import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Text,
  NumberInputProps,
} from '@chakra-ui/react';

interface Props extends NumberInputProps {
  styleClass?: string;
  label?: string;
  width?: string | number;
  showStepper?: boolean;
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
    ...rest
  } = props;
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
        keepWithinRange={false}
        clampValueOnBlur={true}
        className="number-input-field"
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
