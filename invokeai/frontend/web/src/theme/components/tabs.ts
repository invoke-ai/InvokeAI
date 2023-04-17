import { tabsAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

import { isMobile } from 'theme/util/isMobile';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIRoot = defineStyle((_props) => {
  return {
    display: 'flex',
    columnGap: 4,
  };
});

const invokeAIRootMobile = defineStyle((_props) => {
  return {
    position: 'relative',
    display: 'block',
  };
});

const invokeAITab = defineStyle((_props) => ({}));

const invokeAITablist = defineStyle((_props) => ({
  display: 'flex',
  flexDirection: isMobile ? 'row' : 'column',
  gap: 1,
  color: 'base.700',
  justifyContent: isMobile ? 'center' : '',
  button: {
    fontSize: 'sm',
    padding: 2,
    paddingLeft: isMobile ? '5vw' : '',
    paddingRight: isMobile ? '5vw' : '',
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
  root: isMobile ? invokeAIRootMobile(props) : invokeAIRoot(props),
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
