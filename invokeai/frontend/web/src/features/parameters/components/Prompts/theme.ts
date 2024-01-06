import { defineStyle } from '@chakra-ui/react';

export const buttonVariantPromptOverlay = defineStyle(() => {
  const _disabled = {
    bg: 'none',
    color: 'base.500',
    svg: {
      fill: 'base.500',
    },
    opacity: 0.7,
  };

  return {
    fontSize: 14,
    h: 'min-content',
    w: 'min-content',
    minW: 'unset',
    minH: 'unset',
    bg: 'none',
    color: 'base.400',
    svg: {
      fill: 'base.400',
    },
    _disabled,
    _hover: {
      bg: 'none',
      color: 'base.100',
      svg: {
        fill: 'base.100',
      },
      _disabled,
    },
    '&[data-checked="true"]': {
      color: 'blue.300',
      svg: {
        fill: 'blue.300',
      },
      _hover: {
        color: 'blue.400',
        svg: {
          fill: 'blue.400',
        },
      },
    },
  };
});
