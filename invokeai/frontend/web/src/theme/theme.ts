import { ThemeOverride } from '@chakra-ui/react';

import { InvokeAIColors } from './colors/colors';
import { accordionTheme } from './components/accordion';
import { buttonTheme } from './components/button';
import { checkboxTheme } from './components/checkbox';
import { editableTheme } from './components/editable';
import { formLabelTheme } from './components/formLabel';
import { inputTheme } from './components/input';
import { menuTheme } from './components/menu';
import { modalTheme } from './components/modal';
import { numberInputTheme } from './components/numberInput';
import { popoverTheme } from './components/popover';
import { progressTheme } from './components/progress';
import { no_scrollbar } from './components/scrollbar';
import { selectTheme } from './components/select';
import { skeletonTheme } from './components/skeleton';
import { sliderTheme } from './components/slider';
import { switchTheme } from './components/switch';
import { tabsTheme } from './components/tabs';
import { textTheme } from './components/text';
import { textareaTheme } from './components/textarea';
import { tooltipTheme } from './components/tooltip';
import { reactflowStyles } from './custom/reactflow';

export const theme: ThemeOverride = {
  config: {
    cssVarPrefix: 'invokeai',
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  layerStyles: {
    body: {
      bg: 'base.50',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.900', color: 'base.50' },
    },
    first: {
      bg: 'base.100',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.850', color: 'base.100' },
    },
    second: {
      bg: 'base.200',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.800', color: 'base.100' },
    },
    third: {
      bg: 'base.300',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.750', color: 'base.100' },
    },
    nodeBody: {
      bg: 'base.100',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.800', color: 'base.100' },
    },
    nodeHeader: {
      bg: 'base.200',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.900', color: 'base.100' },
    },
    nodeFooter: {
      bg: 'base.200',
      color: 'base.900',
      '.chakra-ui-dark &': { bg: 'base.900', color: 'base.100' },
    },
  },
  styles: {
    global: () => ({
      layerStyle: 'body',
      '*': { ...no_scrollbar },
      ...reactflowStyles,
    }),
  },
  direction: 'ltr',
  fonts: {
    body: `'Inter Variable', sans-serif`,
  },
  shadows: {
    light: {
      accent: `0 0 10px 0 var(--invokeai-colors-accent-300)`,
      accentHover: `0 0 10px 0 var(--invokeai-colors-accent-400)`,
      ok: `0 0 7px var(--invokeai-colors-ok-600)`,
      working: `0 0 7px var(--invokeai-colors-working-600)`,
      error: `0 0 7px var(--invokeai-colors-error-600)`,
    },
    dark: {
      accent: `0 0 10px 0 var(--invokeai-colors-accent-600)`,
      accentHover: `0 0 10px 0 var(--invokeai-colors-accent-500)`,
      ok: `0 0 7px var(--invokeai-colors-ok-400)`,
      working: `0 0 7px var(--invokeai-colors-working-400)`,
      error: `0 0 7px var(--invokeai-colors-error-400)`,
    },
    selected: {
      light:
        '0px 0px 0px 1px var(--invokeai-colors-base-150), 0px 0px 0px 4px var(--invokeai-colors-accent-400)',
      dark: '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 4px var(--invokeai-colors-accent-500)',
    },
    hoverSelected: {
      light:
        '0px 0px 0px 1px var(--invokeai-colors-base-150), 0px 0px 0px 4px var(--invokeai-colors-accent-500)',
      dark: '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 4px var(--invokeai-colors-accent-400)',
    },
    hoverUnselected: {
      light:
        '0px 0px 0px 1px var(--invokeai-colors-base-150), 0px 0px 0px 3px var(--invokeai-colors-accent-500)',
      dark: '0px 0px 0px 1px var(--invokeai-colors-base-900), 0px 0px 0px 3px var(--invokeai-colors-accent-400)',
    },
    nodeSelected: {
      light: `0 0 0 3px var(--invokeai-colors-accent-400)`,
      dark: `0 0 0 3px var(--invokeai-colors-accent-500)`,
    },
    nodeHovered: {
      light: `0 0 0 2px var(--invokeai-colors-accent-500)`,
      dark: `0 0 0 2px var(--invokeai-colors-accent-400)`,
    },
    nodeHoveredSelected: {
      light: `0 0 0 3px var(--invokeai-colors-accent-500)`,
      dark: `0 0 0 3px var(--invokeai-colors-accent-400)`,
    },
    nodeInProgress: {
      light:
        '0 0 0 2px var(--invokeai-colors-accent-500), 0 0 10px 2px var(--invokeai-colors-accent-600)',
      dark: '0 0 0 2px var(--invokeai-colors-yellow-400), 0 0 20px 2px var(--invokeai-colors-orange-700)',
    },
  },
  colors: InvokeAIColors,
  components: {
    Button: buttonTheme, // Button and IconButton
    Input: inputTheme,
    Editable: editableTheme,
    Textarea: textareaTheme,
    Tabs: tabsTheme,
    Progress: progressTheme,
    Accordion: accordionTheme,
    FormLabel: formLabelTheme,
    Switch: switchTheme,
    NumberInput: numberInputTheme,
    Select: selectTheme,
    Skeleton: skeletonTheme,
    Slider: sliderTheme,
    Popover: popoverTheme,
    Modal: modalTheme,
    Checkbox: checkboxTheme,
    Menu: menuTheme,
    Text: textTheme,
    Tooltip: tooltipTheme,
  },
};
