import { switchAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAITrack = defineStyle((props) => {
  const { colorScheme: c } = props;

  return {
    bg: 'base.600',

    _focusVisible: {
      boxShadow: 'none',
    },
    _checked: {
      bg: `${c}.600`,
    },
  };
});

const invokeAIThumb = defineStyle((props) => {
  const { colorScheme: c } = props;

  return {
    bg: `${c}.50`,
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
    colorScheme: 'accent',
  },
});
