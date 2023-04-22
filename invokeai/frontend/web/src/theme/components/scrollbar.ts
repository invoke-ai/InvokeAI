import { ChakraProps } from '@chakra-ui/react';

export const no_scrollbar: ChakraProps['sx'] = {
  '::-webkit-scrollbar': {
    display: 'none',
  },
  scrollbarWidth: 'none',
};

export const scrollbar: ChakraProps['sx'] = {
  scrollbarColor: 'accent.600 transparent',
  scrollbarWidth: 'thick',
  '::-webkit-scrollbar': {
    width: '6px', // Vertical Scrollbar Width
    height: '6px', // Horizontal Scrollbar Height
  },
  '::-webkit-scrollbar-track': {
    background: 'transparent',
  },
  '::-webkit-scrollbar-thumb': {
    background: 'accent.600',
    borderRadius: '8px',
    borderWidth: '4px',
    borderColor: 'accent.600',
  },
  '::-webkit-scrollbar-thumb:hover': {
    background: 'accent.500',
    borderColor: 'accent.500',
  },
  '::-webkit-scrollbar-button': {
    background: 'transparent',
  },
};
