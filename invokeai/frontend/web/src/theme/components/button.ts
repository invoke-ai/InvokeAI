import { defineStyle, defineStyleConfig } from '@chakra-ui/react';

const invokeAI = defineStyle((props) => {
  const { colorScheme: c } = props;
  // must specify `_disabled` colors if we override `_hover`, else hover on disabled has no styles
  const _disabled = {
    bg: `${c}.600`,
    color: `${c}.100`,
    svg: {
      fill: `${c}.100`,
    },
  };

  return {
    bg: `${c}.700`,
    color: `${c}.100`,
    borderRadius: 'base',
    svg: {
      fill: `${c}.100`,
    },
    _disabled,
    _hover: {
      bg: `${c}.650`,
      color: `${c}.50`,
      svg: {
        fill: `${c}.50`,
      },
      _disabled,
    },
    _checked: {
      bg: 'accent.700',
      color: 'accent.100',
      svg: {
        fill: 'accent.100',
      },
      _disabled,
      _hover: {
        bg: 'accent.600',
        color: 'accent.50',
        svg: {
          fill: 'accent.50',
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
