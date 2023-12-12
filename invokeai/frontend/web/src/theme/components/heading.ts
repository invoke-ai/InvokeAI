import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const accent = defineStyle((props) => ({
  color: mode('accent.500', 'accent.300')(props),
}));

export const headingTheme = defineStyleConfig({
  variants: {
    accent,
  },
});
