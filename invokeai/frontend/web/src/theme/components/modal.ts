import { modalAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIOverlay = defineStyle({
  bg: 'blackAlpha.600',
});

const invokeAIDialogContainer = defineStyle({});

const invokeAIDialog = defineStyle((_props) => {
  return {
    bg: 'base.850',
    maxH: '80vh',
  };
});

const invokeAIHeader = defineStyle((_props) => {
  return {
    fontWeight: '600',
    fontSize: 'lg',
    color: 'base.200',
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
