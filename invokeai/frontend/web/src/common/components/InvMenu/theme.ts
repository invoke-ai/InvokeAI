import { menuAnatomy } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/react';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(menuAnatomy.keys);

// define the base component styles
const invokeAI = definePartsStyle(() => ({
  // define the part you're going to style
  button: {
    // this will style the MenuButton component
    bg: 'base.500',
    color: 'base.100',
    _hover: {
      bg: 'base.600',
      color: 'base.50',
      fontWeight: 'semibold',
    },
  },
  list: {
    zIndex: 9999,
    color: 'base.150',
    bg: 'base.800',
    shadow: 'dark-lg',
    border: 'none',
    p: 1,
  },
  item: {
    // this will style the MenuItem and MenuItemOption components
    borderRadius: 'sm',
    fontSize: 'sm',
    bg: 'base.800',
    _hover: {
      bg: 'base.700',
      svg: {
        opacity: 1,
      },
    },
    _focus: {
      bg: 'base.700',
    },
    svg: {
      opacity: 0.7,
      fontSize: 14,
    },
    "&[data-destructive='true']": {
      color: 'error.300',
      fill: 'error.300',
      _hover: {
        bg: 'error.600',
        color: 'base.50',
        fill: 'base.50',
      },
    },
    "&[aria-selected='true']": {
      fontWeight: 'semibold',
      bg: 'blue.300 !important',
      color: 'base.800 !important',
      _hover: {
        color: 'base.900 !important',
        bg: 'blue.400 !important',
      },
    },
    "&[aria-selected='true'] [data-option-desc='true']": {
      color: 'base.800',
    },
  },
  divider: {
    borderColor: 'base.700',
  },
  groupTitle: {
    m: 0,
    px: 3,
    py: 2,
    color: 'base.500',
  },
}));

export const menuTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
