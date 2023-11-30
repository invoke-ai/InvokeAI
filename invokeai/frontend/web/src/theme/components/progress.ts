import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIFilledTrack = defineStyle((_props) => ({
  bg: 'accentAlpha.700',
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
