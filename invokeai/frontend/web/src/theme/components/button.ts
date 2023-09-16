import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const invokeAI = defineStyle((props) => {
  const { colorScheme: c } = props;
  // must specify `_disabled` colors if we override `_hover`, else hover on disabled has no styles

  if (c === 'base') {
    const _disabled = {
      bg: mode('base.150', 'base.700')(props),
      color: mode('base.300', 'base.500')(props),
      svg: {
        fill: mode('base.500', 'base.500')(props),
      },
      opacity: 1,
    };

    const data_progress = {
      bg: 'none',
      color: mode('base.300', 'base.500')(props),
      svg: {
        fill: mode('base.500', 'base.500')(props),
      },
      opacity: 1,
    };

    return {
      bg: mode('base.250', 'base.600')(props),
      color: mode('base.850', 'base.100')(props),
      borderRadius: 'base',
      svg: {
        fill: mode('base.850', 'base.100')(props),
      },
      _hover: {
        bg: mode('base.300', 'base.500')(props),
        color: mode('base.900', 'base.50')(props),
        svg: {
          fill: mode('base.900', 'base.50')(props),
        },
        _disabled,
      },
      _disabled,
      '&[data-progress="true"]': { ...data_progress, _hover: data_progress },
    };
  }

  const _disabled = {
    bg: mode(`${c}.250`, `${c}.700`)(props),
    color: mode(`${c}.50`, `${c}.500`)(props),
    svg: {
      fill: mode(`${c}.50`, `${c}.500`)(props),
      filter: 'unset',
    },
    opacity: 1,
    filter: mode(undefined, 'saturate(65%)')(props),
  };

  const data_progress = {
    // bg: 'none',
    color: mode(`${c}.50`, `${c}.500`)(props),
    svg: {
      fill: mode(`${c}.50`, `${c}.500`)(props),
      filter: 'unset',
    },
    opacity: 0.7,
    filter: mode(undefined, 'saturate(65%)')(props),
  };

  return {
    bg: mode(`${c}.400`, `${c}.600`)(props),
    color: mode(`base.50`, `base.100`)(props),
    borderRadius: 'base',
    svg: {
      fill: mode(`base.50`, `base.100`)(props),
    },
    _disabled,
    _hover: {
      bg: mode(`${c}.500`, `${c}.500`)(props),
      color: mode('white', `base.50`)(props),
      svg: {
        fill: mode('white', `base.50`)(props),
      },
      _disabled,
    },
    '&[data-progress="true"]': { ...data_progress, _hover: data_progress },
  };
});

const invokeAIOutline = defineStyle((props) => {
  const { colorScheme: c } = props;
  const borderColor = mode(`gray.200`, `whiteAlpha.300`)(props);
  return {
    border: '1px solid',
    borderColor: c === 'gray' ? borderColor : 'currentColor',
    '.chakra-button__group[data-attached][data-orientation=horizontal] > &:not(:last-of-type)':
      {
        marginEnd: '-1px',
      },
    '.chakra-button__group[data-attached][data-orientation=vertical] > &:not(:last-of-type)':
      {
        marginBottom: '-1px',
      },
  };
});

export const buttonTheme = defineStyleConfig({
  variants: {
    invokeAI,
    invokeAIOutline,
  },
  defaultProps: {
    variant: 'invokeAI',
    colorScheme: 'base',
  },
});
