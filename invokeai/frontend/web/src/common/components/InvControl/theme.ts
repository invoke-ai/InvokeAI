import { formAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
  defineStyleConfig,
} from '@chakra-ui/styled-system';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const formBaseStyle = definePartsStyle((props) => {
  return {
    container: {
      display: 'flex',
      flexDirection: props.orientation === 'vertical' ? 'column' : 'row',
      alignItems: props.orientation === 'vertical' ? 'flex-start' : 'center',
      gap: 4,
      // h: props.orientation === 'vertical' ? 'unset' : 8,
    },
  };
});

const withHelperText = definePartsStyle(() => ({
  container: {
    flexDirection: 'column',
    gap: 0,
    h: 'unset',
    '> div': {
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      gap: 4,
      h: 8,
      w: 'full',
    },
  },
  helperText: {
    w: 'full',
    fontSize: 'sm',
    color: 'base.400',
    m: 0,
  },
}));

export const formTheme = defineMultiStyleConfig({
  baseStyle: formBaseStyle,
  variants: {
    withHelperText,
  },
});

const formLabelBaseStyle = defineStyle(() => {
  return {
    fontSize: 'sm',
    marginEnd: 0,
    mb: 0,
    flexShrink: 0,
    flexGrow: 0,
    fontWeight: 'semibold',
    transitionProperty: 'common',
    transitionDuration: 'normal',
    whiteSpace: 'nowrap',
    userSelect: 'none',
    _disabled: {
      opacity: 0.4,
    },
    color: 'base.300',
    _invalid: {
      color: 'error.300',
    },
  };
});

export const formLabelTheme = defineStyleConfig({
  baseStyle: formLabelBaseStyle,
});
