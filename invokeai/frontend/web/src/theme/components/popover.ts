import { popoverAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { cssVar } from '@chakra-ui/theme-tools';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const $popperBg = cssVar('popper-bg');
const $arrowBg = cssVar('popper-arrow-bg');
const $arrowShadowColor = cssVar('popper-arrow-shadow-color');

const invokeAIContent = defineStyle((props) => {
  return {
    [$arrowBg.variable]: `colors.base.800`,
    [$popperBg.variable]: `colors.base.800`,
    [$arrowShadowColor.variable]: `colors.base.600`,
    minW: 'unset',
    width: 'unset',
    p: 4,
    borderWidth: '2px',
    borderStyle: 'solid',
    borderColor: mode('base.500', 'base.600')(props),
    bg: mode('base.200', 'base.800')(props),
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
