import { switchAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAITrack = defineStyle((_props) => {
  return {
    bg: 'base.600',
    p: 1,
    _focusVisible: {
      boxShadow: 'none',
    },
    _checked: {
      bg: 'blue.500',
    },
  };
});

const invokeAIThumb = defineStyle((_props) => {
  return {
    bg: 'base.50',
  };
});

const invokeAI = definePartsStyle((props) => ({
  container: {},
  track: invokeAITrack(props),
  thumb: invokeAIThumb(props),
}));

export const switchTheme = defineMultiStyleConfig({
  variants: { invokeAI },
  defaultProps: {
    size: 'md',
    variant: 'invokeAI',
  },
});
