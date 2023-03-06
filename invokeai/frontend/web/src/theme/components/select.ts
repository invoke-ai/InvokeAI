import { selectAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';
import { getInputOutlineStyles } from '../util/getInputOutlineStyles';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIIcon = defineStyle((_props) => {
  return {
    color: 'base.300',
  };
});

const invokeAIField = defineStyle((props) => ({
  fontWeight: '600',
  ...getInputOutlineStyles(props),
}));

const invokeAI = definePartsStyle((props) => {
  return {
    field: invokeAIField(props),
    icon: invokeAIIcon(props),
  };
});

export const selectTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    size: 'sm',
    variant: 'invokeAI',
  },
});
