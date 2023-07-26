import { menuAnatomy } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';
import { MotionProps } from 'framer-motion';

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
    color: mode('base.900', 'base.150')(props),
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
      svg: {
        opacity: 1,
      },
    },
    _focus: {
      bg: mode('base.400', 'base.600')(props),
    },
    svg: {
      opacity: 0.7,
      fontSize: 14,
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

export const menuListMotionProps: MotionProps = {
  variants: {
    enter: {
      visibility: 'visible',
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.07,
        ease: [0.4, 0, 0.2, 1],
      },
    },
    exit: {
      transitionEnd: {
        visibility: 'hidden',
      },
      opacity: 0,
      scale: 0.8,
      transition: {
        duration: 0.07,
        easings: 'easeOut',
      },
    },
  },
};
