import {
  forwardRef,
  Icon as ChakraIcon,
  NumberDecrementStepper as ChakraNumberDecrementStepper,
  NumberIncrementStepper as ChakraNumberIncrementStepper,
  NumberInputStepper as ChakraNumberInputStepper,
} from '@chakra-ui/react';
import { memo } from 'react';
import { FaMinus, FaPlus } from 'react-icons/fa6';

import type { InvNumberInputStepperProps } from './types';

export const InvNumberInputStepper = memo(
  forwardRef<InvNumberInputStepperProps, typeof ChakraNumberInputStepper>(
    (props: InvNumberInputStepperProps, ref) => {
      return (
        <ChakraNumberInputStepper ref={ref} {...props}>
          <ChakraNumberIncrementStepper>
            <ChakraIcon as={FaPlus} boxSize={2} />
          </ChakraNumberIncrementStepper>
          <ChakraNumberDecrementStepper>
            <ChakraIcon as={FaMinus} boxSize={2} />
          </ChakraNumberDecrementStepper>
        </ChakraNumberInputStepper>
      );
    }
  )
);

InvNumberInputStepper.displayName = 'InvNumberInputStepper';
