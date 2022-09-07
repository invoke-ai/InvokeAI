import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  HStack,
  Text,
  FormLabel,
} from '@chakra-ui/react';

type Props = {
  label: string;
  value: number;
  onChange: (valueAsString: string, valueAsNumber: number) => void;
  step?: number;
  min?: number;
  max?: number;
  precision?: number;
  isDisabled?: boolean;
};

const SDNumberInput = ({
  label,
  value,
  onChange,
  step,
  min,
  max,
  precision,
  isDisabled = false,
}: Props) => {
  return (
    <FormControl isDisabled={isDisabled}>
      <HStack>
        <FormLabel marginInlineEnd={0} marginBottom={1}>
          <Text fontSize={'sm'} whiteSpace='nowrap'>
            {label}
          </Text>
        </FormLabel>
        <NumberInput
          size={'sm'}
          step={step}
          min={min}
          max={max}
          precision={precision}
          onChange={onChange}
          value={value}
        >
          <NumberInputField />
          <NumberInputStepper>
            <NumberIncrementStepper />
            <NumberDecrementStepper />
          </NumberInputStepper>
        </NumberInput>
      </HStack>
    </FormControl>
  );
};

export default SDNumberInput;
