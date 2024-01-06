import { sliderAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const container = defineStyle(() => ({}));

const track = defineStyle(() => {
  return {
    bg: 'base.600',
    h: 2,
  };
});

const filledTrack = defineStyle((_props) => {
  return {
    bg: 'base.400',
    h: 2,
  };
});

const thumb = defineStyle(() => {
  return {
    w: 4,
    h: 4,
    bg: 'base.400',
    borderRadius: 'base',
    borderColor: 'base.200',
    borderWidth: 2,
    _hover: {
      transform: `translateY(-50%) scale(1.15)`,
      transition: 'transform 0.1s',
      _active: {
        transform: `translateY(-50%) scale(1.22)`,
        transition: 'transform 0.05s',
      },
    },
  };
});

const mark = defineStyle(() => {
  return {
    fontSize: '10px',
    color: 'base.400',
    mt: 2,
  };
});

const baseStyle = definePartsStyle((props) => ({
  container: container(),
  track: track(),
  filledTrack: filledTrack(props),
  thumb: thumb(),
  mark: mark(),
}));

export const sliderTheme = defineMultiStyleConfig({
  baseStyle,
  defaultProps: {
    colorScheme: 'base',
  },
});
