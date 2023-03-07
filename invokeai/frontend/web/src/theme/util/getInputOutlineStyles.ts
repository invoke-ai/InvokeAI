import { StyleFunctionProps } from '@chakra-ui/theme-tools';

export const getInputOutlineStyles = (_props: StyleFunctionProps) => ({
  outline: 'none',
  borderWidth: 2,
  borderStyle: 'solid',
  borderColor: 'base.800',
  bg: 'base.900',
  borderRadius: 'base',
  color: 'base.100',
  boxShadow: 'none',
  _hover: {
    borderColor: 'base.600',
  },
  _focus: {
    borderColor: 'accent.700',
    boxShadow: 'none',
    _hover: {
      borderColor: 'accent.600',
    },
  },
  _invalid: {
    borderColor: 'error.700',
    boxShadow: 'none',
    _hover: {
      borderColor: 'error.600',
    },
  },
  _disabled: {
    borderColor: 'base.700',
    bg: 'base.700',
    color: 'base.400',
    _hover: {
      borderColor: 'base.700',
    },
  },
  _placeholder: {
    color: 'base.400',
  },
});
