import { StyleFunctionProps, mode } from '@chakra-ui/theme-tools';

export const getInputOutlineStyles = (props: StyleFunctionProps) => ({
  outline: 'none',
  borderWidth: 2,
  borderStyle: 'solid',
  borderColor: mode('base.200', 'base.800')(props),
  bg: mode('base.50', 'base.900')(props),
  borderRadius: 'base',
  color: mode('base.900', 'base.100')(props),
  boxShadow: 'none',
  _hover: {
    borderColor: mode('base.300', 'base.600')(props),
  },
  _focus: {
    borderColor: mode('accent.200', 'accent.600')(props),
    boxShadow: 'none',
    _hover: {
      borderColor: mode('accent.300', 'accent.500')(props),
    },
  },
  _invalid: {
    borderColor: mode('error.300', 'error.600')(props),
    boxShadow: 'none',
    _hover: {
      borderColor: mode('error.400', 'error.500')(props),
    },
  },
  _disabled: {
    borderColor: mode('base.300', 'base.700')(props),
    bg: mode('base.300', 'base.700')(props),
    color: mode('base.600', 'base.400')(props),
    _hover: {
      borderColor: mode('base.300', 'base.700')(props),
    },
  },
  _placeholder: {
    color: mode('base.700', 'base.400')(props),
  },
  '::selection': {
    bg: mode('accent.200', 'accent.400')(props),
  },
});
