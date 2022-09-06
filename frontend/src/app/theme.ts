import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

// Dark Mode setup
const config: ThemeConfig = {
  initialColorMode: 'dark',
  useSystemColorMode: false,
};

export const theme = extendTheme({ config });
