import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { cssVar } from '@chakra-ui/theme-tools';

const $arrowBg = cssVar('popper-arrow-bg');

const baseStyle = defineStyle(() => ({
  borderRadius: 'md',
  shadow: 'dark-lg',
  bg: 'base.200',
  color: 'base.800',
  [$arrowBg.variable]: 'colors.base.200',
  pt: 1,
  px: 2,
  pb: 1,
}));

// export the component theme
export const tooltipTheme = defineStyleConfig({ baseStyle });
