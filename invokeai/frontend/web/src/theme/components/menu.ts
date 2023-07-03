import { menuAnatomy } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(menuAnatomy.keys);

// define the base component styles
const invokeAI = definePartsStyle((props) => ({
  // define the part you're going to style
  button: {
    // this will style the MenuButton component
    fontWeight: 500,
    bg: mode('base.300', 'base.500')(props),
    color: mode('base.900', 'base.100')(props),
    _hover: {
      bg: mode('base.400', 'base.600')(props),
      color: mode('base.900', 'base.50')(props),
      fontWeight: 600,
    },
  },
  list: {
    zIndex: 9999,
    bg: mode('base.200', 'base.800')(props),
    shadow: 'dark-lg',
    border: 'none',
  },
  item: {
    // this will style the MenuItem and MenuItemOption components
    fontSize: 'sm',
    bg: mode('base.200', 'base.800')(props),
    _hover: {
      bg: mode('base.300', 'base.700')(props),
    },
    _focus: {
      bg: mode('base.400', 'base.600')(props),
    },
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
