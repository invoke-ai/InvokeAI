import type { ThemeOverride, ToastProviderProps } from '@chakra-ui/react';
import { accordionTheme } from 'common/components/InvAccordion/theme';
import { badgeTheme } from 'common/components/InvBadge/theme';
import { buttonTheme } from 'common/components/InvButton/theme';
import { cardTheme } from 'common/components/InvCard/theme';
import { checkboxTheme } from 'common/components/InvCheckbox/theme';
import {
  formErrorTheme,
  formLabelTheme,
  formTheme,
} from 'common/components/InvControl/theme';
import { editableTheme } from 'common/components/InvEditable/theme';
import { headingTheme } from 'common/components/InvHeading/theme';
import { inputTheme } from 'common/components/InvInput/theme';
import { menuTheme } from 'common/components/InvMenu/theme';
import { modalTheme } from 'common/components/InvModal/theme';
import { numberInputTheme } from 'common/components/InvNumberInput/theme';
import { popoverTheme } from 'common/components/InvPopover/theme';
import { progressTheme } from 'common/components/InvProgress/theme';
import { skeletonTheme } from 'common/components/InvSkeleton/theme';
import { sliderTheme } from 'common/components/InvSlider/theme';
import { switchTheme } from 'common/components/InvSwitch/theme';
import { tabsTheme } from 'common/components/InvTabs/theme';
import { textTheme } from 'common/components/InvText/theme';
import { textareaTheme } from 'common/components/InvTextarea/theme';
import { tooltipTheme } from 'common/components/InvTooltip/theme';
import { resizeHandleTheme } from 'features/ui/components/tabs/ResizeHandle';

import {
  InvokeAIColors,
  layerStyleBody,
  layerStyleDanger,
  layerStyleFirst,
  layerStyleNodeBody,
  layerStyleNodeFooter,
  layerStyleNodeHeader,
  layerStyleSecond,
  layerStyleThird,
} from './colors';
import { reactflowStyles } from './reactflow';
import { no_scrollbar } from './scrollbar';
import { space } from './space';

export const theme: ThemeOverride = {
  config: {
    cssVarPrefix: 'invokeai',
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  layerStyles: {
    body: layerStyleBody,
    first: layerStyleFirst,
    second: layerStyleSecond,
    third: layerStyleThird,
    nodeBody: layerStyleNodeBody,
    nodeHeader: layerStyleNodeHeader,
    nodeFooter: layerStyleNodeFooter,
    danger: layerStyleDanger,
  },
  styles: {
    global: () => ({
      body: { bg: 'base.900', color: 'base.50' },
      '*': { ...no_scrollbar },
      ...reactflowStyles,
    }),
  },
  radii: {
    base: '4px',
    lg: '8px',
    md: '4px',
    sm: '2px',
  },
  direction: 'ltr',
  fonts: {
    body: "'Inter Variable', sans-serif",
    heading: "'Inter Variable', sans-serif",
  },
  shadows: {
    blue: '0 0 10px 0 var(--invokeai-colors-blue-600)',
    blueHover: '0 0 10px 0 var(--invokeai-colors-blue-500)',
    ok: '0 0 7px var(--invokeai-colors-ok-400)',
    working: '0 0 7px var(--invokeai-colors-working-400)',
    error: '0 0 7px var(--invokeai-colors-error-400)',
    selected:
      '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 4px var(--invokeai-colors-blue-500)',
    hoverSelected:
      '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 4px var(--invokeai-colors-blue-400)',
    hoverUnselected:
      '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 3px var(--invokeai-colors-blue-400)',
    nodeSelected: '0 0 0 3px var(--invokeai-colors-blue-500)',
    nodeHovered: '0 0 0 2px var(--invokeai-colors-blue-400)',
    nodeHoveredSelected: '0 0 0 3px var(--invokeai-colors-blue-400)',
    nodeInProgress:
      '0 0 0 2px var(--invokeai-colors-yellow-400), 0 0 20px 2px var(--invokeai-colors-orange-700)',
  },
  colors: InvokeAIColors,
  components: {
    Accordion: accordionTheme,
    Badge: badgeTheme,
    Button: buttonTheme,
    Card: cardTheme,
    Checkbox: checkboxTheme,
    Editable: editableTheme,
    Form: formTheme,
    FormLabel: formLabelTheme,
    Heading: headingTheme,
    Input: inputTheme,
    Menu: menuTheme,
    Modal: modalTheme,
    NumberInput: numberInputTheme,
    Popover: popoverTheme,
    Progress: progressTheme,
    Skeleton: skeletonTheme,
    Slider: sliderTheme,
    Switch: switchTheme,
    Tabs: tabsTheme, // WIP
    Text: textTheme,
    Textarea: textareaTheme,
    Tooltip: tooltipTheme,
    FormError: formErrorTheme,
    ResizeHandle: resizeHandleTheme,
  },
  space: space,
  sizes: space,
  fontSizes: {
    xs: '0.65rem',
    sm: '0.75rem',
    md: '0.9rem',
    lg: '1.025rem',
    xl: '1.15rem',
    '2xl': '1.3rem',
    '3xl': '1.575rem',
    '4xl': '1.925rem',
    '5xl': '2.5rem',
    '6xl': '3.25rem',
    '7xl': '4rem',
    '8xl': '6rem',
    '9xl': '8rem',
  },
  // fontSizes: {
  //   xs: '0.5rem',
  //   sm: '0.75rem',
  //   md: '0.875rem',
  //   lg: '1rem',
  //   xl: '1.125rem',
  //   '2xl': '1.25rem',
  //   '3xl': '1.5rem',
  //   '4xl': '1.875rem',
  //   '5xl': '2.25rem',
  //   '6xl': '3rem',
  //   '7xl': '3.75rem',
  //   '8xl': '4.5rem',
  //   '9xl': '6rem',
  // },
};

export const TOAST_OPTIONS: ToastProviderProps = {
  defaultOptions: { isClosable: true, position: 'bottom-right' },
};
