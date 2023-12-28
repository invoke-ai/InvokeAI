import type {
  NumberDecrementStepperProps as ChakraNumberDecrementStepperProps,
  NumberIncrementStepperProps as ChakraNumberIncrementStepperProps,
  NumberInputFieldProps as ChakraNumberInputFieldProps,
  NumberInputProps as ChakraNumberInputProps,
  NumberInputStepperProps as ChakraNumberInputStepperProps,
} from '@chakra-ui/react';

export type InvNumberInputFieldProps = ChakraNumberInputFieldProps;
export type InvNumberInputStepperProps = ChakraNumberInputStepperProps;
export type InvNumberIncrementStepperProps = ChakraNumberIncrementStepperProps;
export type InvNumberDecrementStepperProps = ChakraNumberDecrementStepperProps;

export type InvNumberInputProps = Omit<
  ChakraNumberInputProps,
  'onChange' | 'min' | 'max'
> & {
  /**
   * The value
   */
  value: number;
  /**
   * The minimum value
   */
  min: number;
  /**
   * The maximum value
   */
  max: number;
  /**
   * The default step
   */
  step?: number;
  /**
   * The fine step (used when shift is pressed)
   */
  fineStep?: number;
  /**
   * The change handler
   */
  onChange: (v: number) => void;
  /**
   * Override props for the Chakra NumberInputField component
   */
  numberInputFieldProps?: InvNumberInputFieldProps;
};
