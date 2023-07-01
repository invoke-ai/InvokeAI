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
        bg: mode('accent.400', 'accent.600')(props),
        color: mode('base.50', 'base.100')(props),
        svg: {
          fill: mode(`base.50`, `base.100`)(props),
          filter: mode(
            `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-${c}-600))`,
            `drop-shadow(0px 0px 0.3rem var(--invokeai-colors-${c}-800))`
          )(props),
        },
        _hover: {
          bg: mode('accent.500', 'accent.500')(props),
          color: mode('white', 'base.50')(props),
          svg: {
            fill: mode('white', 'base.50')(props),
          },
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
    colorScheme: 'accent',
  },
});
