import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { getInputFilledStyles } from 'theme/util/getInputFilledStyles';

export const textareaTheme = defineStyleConfig({
  variants: {
    filled: defineStyle((props) => getInputFilledStyles(props)),
    darkFilled: defineStyle((props) => getInputFilledStyles(props)),
  },
  defaultProps: {
    size: 'md',
    variant: 'filled',
  },
});
