import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIFilledTrack = defineStyle((_props) => ({
  bg: 'invokeYellow.500',
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
  baseStyle: {
    track: { borderRadius: '2px' },
  },
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
