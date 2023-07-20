import { editableAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const baseStylePreview = defineStyle({
  borderRadius: 'md',
  py: '1',
  transitionProperty: 'common',
  transitionDuration: 'normal',
});

const baseStyleInput = defineStyle((props) => ({
  borderRadius: 'md',
  py: '1',
  transitionProperty: 'common',
  transitionDuration: 'normal',
  width: 'full',
  _focusVisible: { boxShadow: 'outline' },
  _placeholder: { opacity: 0.6 },
  '::selection': {
    color: mode('accent.900', 'accent.50')(props),
    bg: mode('accent.200', 'accent.400')(props),
  },
}));

const baseStyleTextarea = defineStyle({
  borderRadius: 'md',
  py: '1',
  transitionProperty: 'common',
  transitionDuration: 'normal',
  width: 'full',
  _focusVisible: { boxShadow: 'outline' },
  _placeholder: { opacity: 0.6 },
});

const invokeAI = definePartsStyle((props) => ({
  preview: baseStylePreview,
  input: baseStyleInput(props),
  textarea: baseStyleTextarea,
}));

export const editableTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    size: 'sm',
    variant: 'invokeAI',
  },
});
