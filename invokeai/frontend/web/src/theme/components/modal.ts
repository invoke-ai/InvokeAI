import { modalAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIOverlay = defineStyle((props) => ({
  bg: mode('blackAlpha.700', 'blackAlpha.700')(props),
}));

const invokeAIDialogContainer = defineStyle({});

const invokeAIDialog = defineStyle(() => {
  return {
    layerStyle: 'first',
    maxH: '80vh',
  };
});

const invokeAIHeader = defineStyle(() => {
  return {
    fontWeight: '600',
    fontSize: 'lg',
    layerStyle: 'first',
    borderTopRadius: 'base',
    borderInlineEndRadius: 'base',
  };
});

const invokeAICloseButton = defineStyle({});

const invokeAIBody = defineStyle({
  overflowY: 'scroll',
});

const invokeAIFooter = defineStyle({});

export const invokeAI = definePartsStyle((props) => ({
  overlay: invokeAIOverlay(props),
  dialogContainer: invokeAIDialogContainer,
  dialog: invokeAIDialog(),
  header: invokeAIHeader(),
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
