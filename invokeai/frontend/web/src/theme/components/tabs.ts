import { tabsAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIRoot = defineStyle((_props) => {
  return {
    display: 'flex',
    columnGap: 4,
  };
});

const invokeAITab = defineStyle((_props) => ({}));

const invokeAITablist = defineStyle((props) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: 1,
  color: mode('base.500', 'base.700')(props),
  button: {
    fontSize: 'sm',
    padding: 2,
    borderRadius: 'base',
    _selected: {
      bg: mode('accent.200', 'accent.700')(props),
      color: mode('accent.800', 'accent.100')(props),
      _hover: {
        bg: mode('accent.300', 'accent.600')(props),
        color: mode('accent.900', 'accent.50')(props),
      },
    },
    _hover: {
      bg: mode('base.300', 'base.600')(props),
      color: mode('base.900', 'base.50')(props),
    },
  },
}));

const invokeAITabpanel = defineStyle((_props) => ({
  padding: 0,
  height: '100%',
}));

const invokeAI = definePartsStyle((props) => ({
  root: invokeAIRoot(props),
  tab: invokeAITab(props),
  tablist: invokeAITablist(props),
  tabpanel: invokeAITabpanel(props),
}));

export const tabsTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
