import { tabsAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const appTabsRoot = defineStyle((_props) => {
  return {
    display: 'flex',
    columnGap: 4,
  };
});

const appTabsTab = defineStyle((_props) => ({}));

const appTabsTablist = defineStyle((props) => {
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

const appTabsTabpanel = defineStyle((_props) => ({
  padding: 0,
  height: '100%',
}));

const appTabs = definePartsStyle((props) => ({
  root: appTabsRoot(props),
  tab: appTabsTab(props),
  tablist: appTabsTablist(props),
  tabpanel: appTabsTabpanel(props),
}));

const line = definePartsStyle((props) => ({
  tab: {
    borderTopRadius: 'base',
    px: 4,
    py: 1,
    fontSize: 'sm',
    color: mode('base.600', 'base.400')(props),
    fontWeight: 500,
    _selected: {
      color: mode('accent.600', 'accent.400')(props),
    },
  },
  tabpanel: {
    p: 0,
    pt: 4,
    w: 'full',
    h: 'full',
  },
  tabpanels: {
    w: 'full',
    h: 'full',
  },
}));

export const tabsTheme = defineMultiStyleConfig({
  variants: {
    line,
    appTabs,
  },
  defaultProps: {
    variant: 'appTabs',
    colorScheme: 'accent',
  },
});
