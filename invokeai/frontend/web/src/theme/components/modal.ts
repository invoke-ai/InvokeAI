import { modalAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIOverlay = defineStyle({
  bg: 'blackAlpha.600',
});

const invokeAIDialogContainer = defineStyle({});

const invokeAIDialog = defineStyle((props) => {
  return {
    bg: mode('base.300', 'base.850')(props),
    maxH: '80vh',
  };
});

const invokeAIHeader = defineStyle((props) => {
  return {
    fontWeight: '600',
    fontSize: 'lg',
    color: mode('base.700', 'base.200')(props),
  };
});

const invokeAICloseButton = defineStyle({});

const invokeAIBody = defineStyle({
  overflowY: 'scroll',
});

const invokeAIFooter = defineStyle({});

export const invokeAI = definePartsStyle((props) => ({
  overlay: invokeAIOverlay,
  dialogContainer: invokeAIDialogContainer,
  dialog: invokeAIDialog(props),
  header: invokeAIHeader(props),
  closeButton: invokeAICloseButton,
  body: invokeAIBody,
  footer: invokeAIFooter,
}));

export const modalTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: { variant: 'invokeAI', size: 'lg' },
});
