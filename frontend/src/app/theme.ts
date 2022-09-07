import { extendTheme, type ThemeConfig } from '@chakra-ui/react';
import type { StyleFunctionProps } from '@chakra-ui/styled-system';

export const theme = extendTheme({
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  components: {
    Tooltip: {
      baseStyle: (props: StyleFunctionProps) => ({
        textColor: props.colorMode === 'dark' ? 'gray.800' : 'gray.100',
      }),
    },
  },
});
