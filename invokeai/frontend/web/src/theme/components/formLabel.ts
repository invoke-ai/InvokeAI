import { defineStyle, defineStyleConfig } from '@chakra-ui/styled-system';

const invokeAI = defineStyle((_props) => {
  return {
    fontSize: 'sm',
    marginEnd: 0,
    mb: 1,
    fontWeight: '400',
    transitionProperty: 'common',
    transitionDuration: 'normal',
    whiteSpace: 'nowrap',
    _disabled: {
      opacity: 0.4,
    },
    color: 'base.300',
  };
});

export const formLabelTheme = defineStyleConfig({
  variants: {
    invokeAI,
  },
  defaultProps: {
    variant: 'invokeAI',
  },
});
