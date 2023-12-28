import type { SystemStyleObject } from '@chakra-ui/styled-system';

export const getInputOutlineStyles = (): SystemStyleObject => ({
  outline: 'none',
  borderWidth: 2,
  borderStyle: 'solid',
  borderColor: 'base.800',
  bg: 'base.900',
  borderRadius: 'md',
  color: 'base.100',
  boxShadow: 'none',
  _hover: {
    borderColor: 'base.600',
  },
  _focus: {
    borderColor: 'blue.600',
    boxShadow: 'none',
    _hover: {
      borderColor: 'blue.500',
    },
  },
  _invalid: {
    borderColor: 'error.600',
    boxShadow: 'none',
    _hover: {
      borderColor: 'error.500',
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
  '::selection': {
    bg: 'blue.400',
  },
});
