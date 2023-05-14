import { popoverAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { cssVar } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const $popperBg = cssVar('popper-bg');
const $arrowBg = cssVar('popper-arrow-bg');
const $arrowShadowColor = cssVar('popper-arrow-shadow-color');

const invokeAIContent = defineStyle((_props) => {
  return {
    [$arrowBg.variable]: `colors.base.800`,
    [$popperBg.variable]: `colors.base.800`,
    [$arrowShadowColor.variable]: `colors.base.600`,
    minW: 'unset',
    width: 'unset',
    p: 4,
    bg: 'base.800',
  };
});

const invokeAI = definePartsStyle((props) => ({
  content: invokeAIContent(props),
}));

export const popoverTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
