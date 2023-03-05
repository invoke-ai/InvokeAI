import { menuAnatomy } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/react';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(menuAnatomy.keys);

// define the base component styles
const invokeAI = definePartsStyle({
  // define the part you're going to style
  button: {
    // this will style the MenuButton component
    fontWeight: '600',
    bg: 'base.500',
    color: 'base.200',
    _hover: {
      bg: 'base.600',
      color: 'white',
    },
  },
  list: {
    zIndex: 9999,
    bg: 'base.800',
  },
  item: {
    // this will style the MenuItem and MenuItemOption components
    fontSize: 'sm',
    bg: 'base.800',
    _hover: {
      bg: 'base.750',
    },
    _focus: {
      bg: 'base.700',
    },
  },
});

export const menuTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
