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

const invokeAITablist = defineStyle((props) => {
  const { colorScheme: c } = props;

  return {
    display: 'flex',
    flexDirection: 'column',
    gap: 1,
    color: mode('base.700', 'base.400')(props),
    button: {
      fontSize: 'sm',
      padding: 2,
      borderRadius: 'base',
      textShadow: mode(
        `0 0 0.3rem var(--invokeai-colors-accent-100)`,
        `0 0 0.3rem var(--invokeai-colors-accent-900)`
      )(props),
      svg: {
        fill: mode('base.700', 'base.300')(props),
      },
      _selected: {
        borderBottomColor: 'base.800',
        bg: mode('accent.200', 'accent.600')(props),
        color: mode('accent.800', 'accent.100')(props),
        _hover: {
          bg: mode('accent.300', 'accent.500')(props),
          color: mode('accent.900', 'accent.50')(props),
        },
        svg: {
          fill: mode('base.900', 'base.50')(props),
          filter: mode(
            `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-accent-100))`,
            `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-accent-900))`
          )(props),
        },
      },
      _hover: {
        bg: mode('base.100', 'base.800')(props),
        color: mode('base.900', 'base.50')(props),
        svg: {
          fill: mode(`base.800`, `base.100`)(props),
        },
      },
    },
  };
});

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
