import { numberInputAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

import { getInputOutlineStyles } from '../util/getInputOutlineStyles';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIRoot = defineStyle((_props) => {
  return {
    height: 8,
  };
});

const invokeAIField = defineStyle((props) => {
  return {
    border: 'none',
    fontWeight: '600',
    height: 'auto',
    py: 1,
    ps: 2,
    pe: 6,
    ...getInputOutlineStyles(props),
  };
});

const invokeAIStepperGroup = defineStyle((_props) => {
  return {
    display: 'flex',
  };
});

const invokeAIStepper = defineStyle((_props) => {
  return {
    border: 'none',
    // expand arrow hitbox
    px: 2,
    py: 0,
    mx: -2,
    my: 0,

    svg: {
      color: 'base.300',
      width: 2.5,
      height: 2.5,
      _hover: {
        color: 'base.50',
      },
    },
  };
});

const invokeAI = definePartsStyle((props) => ({
  root: invokeAIRoot(props),
  field: invokeAIField(props),
  stepperGroup: invokeAIStepperGroup(props),
  stepper: invokeAIStepper(props),
}));

export const numberInputTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    size: 'sm',
    variant: 'invokeAI',
  },
});
