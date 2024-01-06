import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';
import {
  forwardRef,
  NumberDecrementStepper as ChakraNumberDecrementStepper,
  NumberIncrementStepper as ChakraNumberIncrementStepper,
  NumberInputStepper as ChakraNumberInputStepper,
} from '@chakra-ui/react';
import { memo } from 'react';

import type { InvNumberInputStepperProps } from './types';

export const InvNumberInputStepper = memo(
  forwardRef<InvNumberInputStepperProps, typeof ChakraNumberInputStepper>(
    (props: InvNumberInputStepperProps, ref) => {
      return (
        <ChakraNumberInputStepper ref={ref} {...props}>
          <ChakraNumberIncrementStepper>
            <ChevronUpIcon />
          </ChakraNumberIncrementStepper>
          <ChakraNumberDecrementStepper>
            <ChevronDownIcon />
          </ChakraNumberDecrementStepper>
        </ChakraNumberInputStepper>
      );
    }
  )
);

InvNumberInputStepper.displayName = 'InvNumberInputStepper';
