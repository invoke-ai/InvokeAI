import { tabsAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIRoot = defineStyle((_props) => {
  return {
    display: 'flex',
    columnGap: 4,
  };
});

const invokeAITab = defineStyle((_props) => ({}));

const invokeAITablist = defineStyle((_props) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: 1,
  color: 'base.700',
  button: {
    fontSize: 'sm',
    padding: 2,
    borderRadius: 'base',
    _selected: {
      bg: 'accent.700',
      color: 'accent.100',
      _hover: {
        bg: 'accent.600',
        color: 'accent.50',
      },
    },
    _hover: {
      bg: 'base.600',
      color: 'base.50',
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
