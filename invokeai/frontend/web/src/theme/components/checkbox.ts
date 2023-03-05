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
    _checked: {
      bg: mode(`${c}.500`, `${c}.200`)(props),
      borderColor: mode(`${c}.500`, `${c}.200`)(props),
      color: mode('white', 'base.900')(props),

      _hover: {
        bg: mode(`${c}.600`, `${c}.300`)(props),
        borderColor: mode(`${c}.600`, `${c}.300`)(props),
      },

      _disabled: {
        borderColor: mode('base.200', 'transparent')(props),
        bg: mode('base.200', 'whiteAlpha.300')(props),
        color: mode('base.500', 'whiteAlpha.500')(props),
      },
    },

    _indeterminate: {
      bg: mode(`${c}.500`, `${c}.200`)(props),
      borderColor: mode(`${c}.500`, `${c}.200`)(props),
      color: mode('white', 'base.900')(props),
    },

    _disabled: {
      bg: mode('base.100', 'whiteAlpha.100')(props),
      borderColor: mode('base.100', 'transparent')(props),
    },

    _focusVisible: {
      boxShadow: 'outline',
    },

    _invalid: {
      borderColor: mode('red.500', 'red.300')(props),
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
