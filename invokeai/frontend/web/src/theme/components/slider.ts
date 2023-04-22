import { sliderAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAITrack = defineStyle((_props) => {
  return {
    bg: 'base.400',
    h: 1.5,
  };
});

const invokeAIFilledTrack = defineStyle((props) => {
  const { colorScheme: c } = props;
  return {
    bg: `${c}.600`,
    h: 1.5,
  };
});

const invokeAIThumb = defineStyle((_props) => {
  return {
    w: 2,
    h: 4,
  };
});

const invokeAIMark = defineStyle((_props) => {
  return {
    fontSize: 'xs',
    fontWeight: '500',
    color: 'base.200',
    mt: 2,
    insetInlineStart: 'unset',
  };
});

const invokeAI = definePartsStyle((props) => ({
  container: {
    _disabled: {
      opacity: 0.6,
      cursor: 'default',
      pointerEvents: 'none',
    },
  },
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
