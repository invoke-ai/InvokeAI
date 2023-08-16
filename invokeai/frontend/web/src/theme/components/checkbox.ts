import { checkboxAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIControl = defineStyle((props) => {
  const { colorScheme: c } = props;

  return {
    bg: mode('base.200', 'base.700')(props),
    borderColor: mode('base.300', 'base.600')(props),
    color: mode('base.900', 'base.100')(props),

    _checked: {
      bg: mode(`${c}.300`, `${c}.500`)(props),
      borderColor: mode(`${c}.300`, `${c}.500`)(props),
      color: mode(`${c}.900`, `${c}.100`)(props),

      _hover: {
        bg: mode(`${c}.400`, `${c}.500`)(props),
        borderColor: mode(`${c}.400`, `${c}.500`)(props),
      },

      _disabled: {
        borderColor: 'transparent',
        bg: 'whiteAlpha.300',
        color: 'whiteAlpha.500',
      },
    },

    _indeterminate: {
      bg: mode(`${c}.300`, `${c}.600`)(props),
      borderColor: mode(`${c}.300`, `${c}.600`)(props),
      color: mode(`${c}.900`, `${c}.100`)(props),
    },

    _disabled: {
      bg: 'whiteAlpha.100',
      borderColor: 'transparent',
    },

    _focusVisible: {
      boxShadow: 'none',
      outline: 'none',
    },

    _invalid: {
      borderColor: mode('error.600', 'error.300')(props),
    },
  };
});

const invokeAI = definePartsStyle((props) => ({
  control: invokeAIControl(props),
}));

export const checkboxTheme = defineMultiStyleConfig({
  variants: {
    invokeAI: invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
    colorScheme: 'accent',
  },
});
