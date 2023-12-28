import type { ChakraProps } from '@chakra-ui/react';

export const no_scrollbar: ChakraProps['sx'] = {
  '::-webkit-scrollbar': {
    display: 'none',
  },
  scrollbarWidth: 'none',
};

export const scrollbar: ChakraProps['sx'] = {
  scrollbarColor: 'blue.600 transparent',
  scrollbarWidth: 'thick',
  '::-webkit-scrollbar': {
    width: '6px', // Vertical Scrollbar Width
    height: '6px', // Horizontal Scrollbar Height
  },
  '::-webkit-scrollbar-track': {
    background: 'transparent',
  },
  '::-webkit-scrollbar-thumb': {
    background: 'blue.600',
    borderRadius: '8px',
    borderWidth: '4px',
    borderColor: 'blue.600',
  },
  '::-webkit-scrollbar-thumb:hover': {
    background: 'blue.500',
    borderColor: 'blue.500',
  },
  '::-webkit-scrollbar-button': {
    background: 'transparent',
  },
};
