import type {
  StyleFunctionProps,
  SystemStyleObject,
} from '@chakra-ui/styled-system';

export const getInputFilledStyles = (
  props: StyleFunctionProps
): SystemStyleObject => {
  const { variant } = props;

  const bg = variant === 'darkFilled' ? 'base.800' : 'base.700';
  const bgHover = variant === 'darkFilled' ? 'base.750' : 'base.650';
  const error = 'error.600';
  const errorHover = 'error.500';
  const fg = 'base.200';

  const baseColors = {
    color: fg,
    bg: bg,
    borderColor: bg,
  };
  const _invalid = {
    borderColor: error,
    _hover: {
      borderColor: errorHover,
    },
  };
  const _hover = {
    bg: bgHover,
    borderColor: bgHover,
  };
  const _focusVisible = {
    ..._hover,
    _invalid,
  };
  const _disabled = {
    _hover: baseColors,
  };
  return {
    ...baseColors,
    minH: '28px',
    borderWidth: 1,
    borderRadius: 'base',
    outline: 'none',
    boxShadow: 'none',
    _hover,
    _focusVisible,
    _invalid,
    _disabled,
  };
};
