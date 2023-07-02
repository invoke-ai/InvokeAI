import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIFilledTrack = defineStyle((_props) => ({
  bg: 'accent.500',
  // TODO: the animation is nice but looks weird bc it is substantially longer than each step
  // so we get to 100% long before it finishes
  // transition: 'width 0.2s ease-in-out',
  _indeterminate: {
    bgGradient:
      'linear(to-r, transparent 0%, accent.500 50%, transparent 100%);',
  },
}));

const invokeAITrack = defineStyle((_props) => {
  const { colorScheme: c } = _props;
  return {
    bg: mode(`${c}.200`, `${c}.700`)(_props),
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
