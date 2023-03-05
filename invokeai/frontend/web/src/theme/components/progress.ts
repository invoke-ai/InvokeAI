import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIFilledTrack = defineStyle((props) => ({
  bg: mode('accent.400', 'accent.600')(props),
  transition: 'width 0.2s ease-in-out',
  _indeterminate: {
    bgGradient: `linear(to-r, transparent 0%, ${mode(
      'accent.400',
      'accent.600'
    )(props)} 50%, transparent 100%);`,
  },
}));

const invokeAITrack = defineStyle((props) => {
  return {
    bg: mode('base.300', 'base.800')(props),
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
