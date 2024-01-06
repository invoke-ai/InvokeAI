import { cssVar, defineStyle, defineStyleConfig } from '@chakra-ui/react';

const $startColor = cssVar('skeleton-start-color');
const $endColor = cssVar('skeleton-end-color');

const invokeAI = defineStyle({
  borderRadius: 'base',
  maxW: 'full',
  maxH: 'full',
  [$startColor.variable]: 'colors.base.700',
  [$endColor.variable]: 'colors.base.500',
});

export const skeletonTheme = defineStyleConfig({
  variants: { invokeAI },
  defaultProps: {
    variant: 'invokeAI',
  },
});
