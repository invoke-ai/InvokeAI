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
  NumberInputProps,
} from '@chakra-ui/react';

interface Props extends NumberInputProps {
  label: string;
  width?: string | number;
}

const SDNumberInput = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'sm',
    size = 'sm',
    width,
    isInvalid,
  } = props;
  return (
    <FormControl isDisabled={isDisabled} width={width} isInvalid={isInvalid}>
      <HStack>
        <FormLabel marginInlineEnd={0} marginBottom={1}>
          <Text fontSize={fontSize} whiteSpace='nowrap'>
            {label}
          </Text>
        </FormLabel>
        <NumberInput size={size} {...props}>
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
