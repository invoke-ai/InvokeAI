import {
  formAnatomy as formParts,
  formErrorAnatomy as formErrorParts,
} from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
  defineStyleConfig,
} from '@chakra-ui/styled-system';

const {
  definePartsStyle: defineFormPartsStyle,
  defineMultiStyleConfig: defineFormMultiStyleConfig,
} = createMultiStyleConfigHelpers(formParts.keys);

const formBaseStyle = defineFormPartsStyle((props) => ({
  container: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 4,
    h: 'unset',
    '> .invcontrol-label-input-wrapper': {
      display: 'flex',
      flexDirection: props.orientation === 'vertical' ? 'column' : 'row',
      alignItems: props.orientation === 'vertical' ? 'flex-start' : 'center',
      gap: props.orientation === 'vertical' ? 2 : 4,
      minH: 8,
      w: 'full',
      _invalid: {
        color: 'error.300',
      },
    },
  },
  helperText: {
    w: 'full',
    fontSize: 'sm',
    color: 'base.400',
    m: 0,
  },
}));

export const formTheme = defineFormMultiStyleConfig({
  baseStyle: formBaseStyle,
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
    h: 'full',
    alignItems: 'center',
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

const { defineMultiStyleConfig: defineFormErrorMultiStyleConfig } =
  createMultiStyleConfigHelpers(formErrorParts.keys);

export const formErrorTheme = defineFormErrorMultiStyleConfig({
  baseStyle: {
    text: {
      color: 'error.300',
    },
  },
});
