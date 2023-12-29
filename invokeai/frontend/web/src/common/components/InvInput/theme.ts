import { inputAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/styled-system';
import { getInputFilledStyles } from 'theme/util/getInputFilledStyles';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

export const inputTheme = defineMultiStyleConfig({
  variants: {
    filled: definePartsStyle((props) => ({
      field: getInputFilledStyles(props),
    })),
    darkFilled: definePartsStyle((props) => ({
      field: getInputFilledStyles(props),
    })),
  },
  defaultProps: {
    size: 'sm',
    variant: 'filled',
  },
});
