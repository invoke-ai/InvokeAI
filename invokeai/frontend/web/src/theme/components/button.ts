import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const invokeAI = defineStyle((props) => {
  const { colorScheme: c } = props;
  // must specify `_disabled` colors if we override `_hover`, else hover on disabled has no styles
  const _disabled = {
    bg: mode(`${c}.200`, `${c}.600`)(props),
    color: mode(`${c}.700`, `${c}.100`)(props),
    svg: {
      fill: mode(`${c}.700`, `${c}.100`)(props),
    },
  };

  return {
    bg: mode(`${c}.300`, `${c}.700`)(props),
    color: mode(`${c}.800`, `${c}.100`)(props),
    borderRadius: 'base',
    svg: {
      fill: mode(`${c}.800`, `${c}.100`)(props),
    },
    _disabled,
    _hover: {
      bg: mode(`${c}.400`, `${c}.650`)(props),
      color: mode(`black`, `${c}.50`)(props),
      svg: {
        fill: mode(`black`, `${c}.50`)(props),
      },
      _disabled,
    },
    _checked: {
      bg: mode('accent.200', 'accent.700')(props),
      color: mode('accent.800', 'accent.100')(props),
      svg: {
        fill: mode('accent.800', 'accent.100')(props),
      },
      _disabled,
      _hover: {
        bg: mode('accent.300', 'accent.600')(props),
        color: mode('accent.900', 'accent.50')(props),
        svg: {
          fill: mode('accent.900', 'accent.50')(props),
        },
        _disabled,
      },
    },
  };
});

export const buttonTheme = defineStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
    colorScheme: 'base',
  },
});
