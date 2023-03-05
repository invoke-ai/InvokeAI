import { accordionAnatomy as parts } from '@chakra-ui/anatomy';
import {
  createMultiStyleConfigHelpers,
  defineStyle,
} from '@chakra-ui/styled-system';
import { mode } from '@chakra-ui/theme-tools';

const { definePartsStyle, defineMultiStyleConfig } =
  createMultiStyleConfigHelpers(parts.keys);

const invokeAIContainer = defineStyle({
  border: 'none',
  pt: 2,
});

const invokeAIButton = defineStyle((props) => {
  const { colorScheme: c } = props;
  return {
    fontWeight: '600',
    fontSize: 'sm',
    border: 'none',
    borderRadius: 'base',
    bg: mode(`${c}.250`, `${c}.800`)(props),
    color: mode('base.900', 'base.100')(props),
    _hover: {
      bg: mode(`${c}.300`, `${c}.700`)(props),
    },
    _expanded: {
      bg: mode(`${c}.250`, `${c}.750`)(props),
      borderBottomRadius: 'none',
      _hover: {
        bg: mode(`${c}.300`, `${c}.700`)(props),
      },
    },
  };
});

const invokeAIPanel = defineStyle((props) => {
  const { colorScheme: c } = props;
  return {
    bg: mode(`${c}.200`, `${c}.800`)(props),
    borderRadius: 'base',
    borderTopRadius: 'none',
    p: 4,
  };
});

const invokeAIIcon = defineStyle({});

const invokeAI = definePartsStyle((props) => ({
  container: invokeAIContainer,
  button: invokeAIButton(props),
  panel: invokeAIPanel(props),
  icon: invokeAIIcon,
}));

export const accordionTheme = defineMultiStyleConfig({
  variants: { invokeAI },
  defaultProps: {
    variant: 'invokeAI',
    colorScheme: 'base',
  },
});
