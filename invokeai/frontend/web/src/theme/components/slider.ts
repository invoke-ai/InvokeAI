import { sliderAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAITrack = defineStyle((props) => {
  return {
    bg: mode('base.300', 'base.400')(props),
    h: 1.5,
  };
});

const invokeAIFilledTrack = defineStyle((props) => {
  const { colorScheme: c } = props;
  return {
    bg: mode(`${c}.400`, `${c}.600`)(props),
    h: 1.5,
  };
});

const invokeAIThumb = defineStyle((_props) => {
  return {
    w: 2,
    h: 4,
  };
});

const invokeAIMark = defineStyle((props) => {
  return {
    fontSize: 'xs',
    fontWeight: '500',
    color: mode('base.800', 'base.200')(props),
    mt: 2,
    insetInlineStart: 'unset',
  };
});

const invokeAI = definePartsStyle((props) => ({
  track: invokeAITrack(props),
  filledTrack: invokeAIFilledTrack(props),
  thumb: invokeAIThumb(props),
  mark: invokeAIMark(props),
}));

export const sliderTheme = defineMultiStyleConfig({
  variants: { invokeAI },
  defaultProps: {
    variant: 'invokeAI',
    colorScheme: 'accent',
  },
});
