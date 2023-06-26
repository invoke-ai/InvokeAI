import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const invokeAI = defineStyle((props) => {
  const { colorScheme: c } = props;
  // must specify `_disabled` colors if we override `_hover`, else hover on disabled has no styles
  const _disabled = {
    bg: mode(`${c}.350`, `${c}.700`)(props),
    color: mode(`${c}.750`, `${c}.150`)(props),
    svg: {
      fill: mode(`${c}.750`, `${c}.150`)(props),
    },
    opacity: 1,
    filter: 'saturate(65%)',
  };

  return {
    bg: mode(`${c}.200`, `${c}.600`)(props),
    color: mode(`${c}.850`, `${c}.100`)(props),
    borderRadius: 'base',
    textShadow: mode(
      `0 0 0.3rem var(--invokeai-colors-${c}-50)`,
      `0 0 0.3rem var(--invokeai-colors-${c}-900)`
    )(props),
    svg: {
      fill: mode(`${c}.850`, `${c}.100`)(props),
      filter: mode(
        `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-${c}-100))`,
        `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-${c}-800))`
      )(props),
    },
    _disabled,
    _hover: {
      bg: mode(`${c}.300`, `${c}.500`)(props),
      color: mode(`${c}.900`, `${c}.50`)(props),
      svg: {
        fill: mode(`${c}.900`, `${c}.50`)(props),
      },
      _disabled,
    },
    _checked: {
      bg: mode('accent.200', 'accent.600')(props),
      color: mode('accent.800', 'accent.100')(props),
      svg: {
        fill: mode('accent.800', 'accent.100')(props),
      },
      _disabled,
      _hover: {
        bg: mode('accent.300', 'accent.500')(props),
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
