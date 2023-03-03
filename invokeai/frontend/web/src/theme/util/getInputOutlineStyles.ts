import { mode, StyleFunctionProps } from '@chakra-ui/theme-tools';

export const getInputOutlineStyles = (props: StyleFunctionProps) => ({
  outline: 'none',
  borderWidth: 2,
  borderStyle: 'solid',
  borderColor: mode('base.300', 'base.800')(props),
  bg: mode('base.200', 'base.900')(props),
  borderRadius: 'base',
  color: mode('base.900', 'base.100')(props),
  boxShadow: 'none',
  _hover: {
    borderColor: mode('base.500', 'base.600')(props),
  },
  _focus: {
    borderColor: mode('accent.600', 'accent.700')(props),
    boxShadow: 'none',
    _hover: {
      borderColor: mode('accent.700', 'accent.600')(props),
    },
  },
  _invalid: {
    borderColor: mode('error.300', 'error.700')(props),
    boxShadow: 'none',
    _hover: {
      borderColor: mode('error.500', 'error.600')(props),
    },
  },
  _disabled: {
    borderColor: mode('base.300', 'base.700')(props),
    bg: mode('base.400', 'base.700')(props),
    color: mode('base.600', 'base.400')(props),
    _hover: {
      borderColor: mode('base.300', 'base.700')(props),
    },
  },
  _placeholder: {
    color: mode('base.600', 'base.400')(props),
  },
});
