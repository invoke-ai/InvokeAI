import { sliderAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers, defineStyle } from '@chakra-ui/react';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const container = defineStyle(() => ({
  h: '28px',
}));

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

const thumb = defineStyle((props) => {
  const { orientation } = props;
  return {
    w: 5,
    h: 5,
    bg: 'base.400',
    borderColor: 'base.200',
    borderWidth: 3,
    _hover: {
      transform:
        orientation === 'vertical'
          ? 'translateX(-50%) scale(1.15)'
          : 'translateY(-50%) scale(1.15)',
      transition: 'transform 0.1s',
      _active: {
        transform:
          orientation === 'vertical'
            ? 'translateX(-50%) scale(1.22)'
            : 'translateY(-50%) scale(1.22)',
        transition: 'transform 0.05s',
      },
    },
  };
});

const mark = defineStyle(() => {
  return {
    fontSize: '10px',
    color: 'base.400',
    mt: 4,
  };
});

const baseStyle = definePartsStyle((props) => ({
  container: container(),
  track: track(),
  filledTrack: filledTrack(props),
  thumb: thumb(props),
  mark: mark(),
}));

export const sliderTheme = defineMultiStyleConfig({
  baseStyle,
  defaultProps: {
    colorScheme: 'base',
  },
});
