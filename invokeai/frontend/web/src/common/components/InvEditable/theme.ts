import { editableAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const baseStylePreview = defineStyle({
  fontSize: 'sm',
  borderRadius: 'md',
  py: '1',
  transitionProperty: 'common',
  transitionDuration: 'normal',
  color: 'base.100',
  _invalid: {
    color: 'error.300'
  }
});

const baseStyleInput = defineStyle(() => ({
  color: 'base.100',
  fontSize: 'sm',
  borderRadius: 'md',
  py: '1',
  transitionProperty: 'common',
  transitionDuration: 'normal',
  width: 'full',
  _focusVisible: { boxShadow: 'none' },
  _placeholder: { opacity: 0.6 },
  '::selection': {
    color: 'blue.900',
    bg: 'blue.300',
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

const invokeAI = definePartsStyle(() => ({
  preview: baseStylePreview,
  input: baseStyleInput(),
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
