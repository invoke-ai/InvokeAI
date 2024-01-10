import { tabsAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const appTabsRoot = defineStyle(() => {
  return {
    display: 'flex',
    gap: 4,
  };
});

const appTabsTab = defineStyle(() => ({}));

const appTabsTablist = defineStyle((_props) => {
  return {
    display: 'flex',
    flexDirection: 'column',
    gap: 1,
  };
});

const appTabsTabpanel = defineStyle(() => ({
  padding: 0,
  height: '100%',
}));

const appTabs = definePartsStyle((props) => ({
  root: appTabsRoot(),
  tab: appTabsTab(),
  tablist: appTabsTablist(props),
  tabpanel: appTabsTabpanel(),
}));

const line = definePartsStyle(() => ({
  tab: {
    borderTopRadius: 'base',
    px: 4,
    py: 1,
    fontSize: 'sm',
    color: 'base.200',
    _selected: {
      color: 'blue.200',
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

const TAB_INACTIVE = 'base.750';
const TAB_ACTIVE = 'base.800';

const collapse = definePartsStyle(() => ({
  indicator: {},
  tablist: {
    bg: TAB_ACTIVE,
    borderRadius: 'base',
    display: 'grid',
    gridAutoFlow: 'column',
    gridAutoColumns: '1fr',
  },
  tab: {
    gap: 4,
    bg: TAB_INACTIVE,
    fontSize: 'sm',
    fontWeight: 'semibold',
    borderInlineEnd: 1,
    borderInlineEndColor: TAB_ACTIVE,
    borderInlineEndStyle: 'solid',
    _first: {
      borderTopStartRadius: 'base',
    },
    // select the tab next to the selected tab
    '&[aria-selected="true"] + button': { borderBottomStartRadius: 'lg' },
    // select the tab prev to the selected tab
    ':has(+ [aria-selected="true"])': { borderBottomEndRadius: 'lg' },
    _selected: {
      bg: TAB_ACTIVE,
    },
    _last: {
      borderTopEndRadius: 'base',
      alignSelf: 'start',
    },
    transitionProperty: 'all',
  },
  tabpanels: { bg: TAB_ACTIVE, borderBottomRadius: 'base' },
  tabpanel: {
    p: 0,
  },
}));

export const tabsTheme = defineMultiStyleConfig({
  variants: {
    line,
    appTabs,
    collapse,
  },
  defaultProps: {
    variant: 'line',
    colorScheme: 'blue',
  },
});
