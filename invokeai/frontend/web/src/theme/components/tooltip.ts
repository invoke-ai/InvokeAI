import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';
import { cssVar } from '@chakra-ui/theme-tools';

const $arrowBg = cssVar('popper-arrow-bg');

// define the base component styles
const baseStyle = defineStyle((props) => ({
  borderRadius: 'base',
  shadow: 'dark-lg',
  bg: mode('base.700', 'base.200')(props),
  [$arrowBg.variable]: mode('colors.base.700', 'colors.base.200')(props),
  pb: 1.5,
}));

// export the component theme
export const tooltipTheme = defineStyleConfig({ baseStyle });
