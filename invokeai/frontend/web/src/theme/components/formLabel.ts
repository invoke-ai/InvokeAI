import { defineStyle, defineStyleConfig } from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const invokeAI = defineStyle((props) => {
  return {
    fontSize: 'sm',
    marginEnd: 0,
    mb: 1,
    fontWeight: '400',
    transitionProperty: 'common',
    transitionDuration: 'normal',
    whiteSpace: 'nowrap',
    _disabled: {
      opacity: 0.4,
    },
    color: mode('base.700', 'base.300')(props),
    _invalid: {
      color: mode('error.500', 'error.300')(props),
    },
  };
});

export const formLabelTheme = defineStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
