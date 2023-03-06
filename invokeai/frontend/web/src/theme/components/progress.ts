import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIFilledTrack = defineStyle((_props) => ({
  bg: 'accent.600',
  transition: 'width 0.2s ease-in-out',
  _indeterminate: {
    bgGradient:
      'linear(to-r, transparent 0%, accent.600 50%, transparent 100%);',
  },
}));

const invokeAITrack = defineStyle((_props) => {
  return {
    bg: 'base.800',
  };
});

const invokeAI = definePartsStyle((props) => ({
  filledTrack: invokeAIFilledTrack(props),
  track: invokeAITrack(props),
}));

export const progressTheme = defineMultiStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
