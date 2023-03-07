import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { getInputOutlineStyles } from '../util/getInputOutlineStyles';

const invokeAI = defineStyle((props) => getInputOutlineStyles(props));

export const textareaTheme = defineStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    size: 'md',
    variant: 'invokeAI',
  },
});
