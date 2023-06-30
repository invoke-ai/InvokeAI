import { selectAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';
import { getInputOutlineStyles } from '../util/getInputOutlineStyles';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIIcon = defineStyle((props) => {
  return {
    color: mode('base.200', 'base.300')(props),
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
