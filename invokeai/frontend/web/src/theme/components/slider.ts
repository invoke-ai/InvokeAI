import { sliderAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAITrack = defineStyle((props) => {
  return {
    bg: mode('base.400', 'base.600')(props),
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

const invokeAIThumb = defineStyle((props) => {
  return {
    w: props.orientation === 'horizontal' ? 2 : 4,
    h: props.orientation === 'horizontal' ? 4 : 2,
    bg: mode('base.50', 'base.100')(props),
  };
});

const invokeAIMark = defineStyle((props) => {
  return {
    fontSize: '2xs',
    fontWeight: '500',
    color: mode('base.700', 'base.400')(props),
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
