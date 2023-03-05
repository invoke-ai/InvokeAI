import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const subtext = defineStyle((props) => ({
  color: mode('base.700', 'base.400')(props),
}));

export const textTheme = defineStyleConfig({
  variants: {
    subtext,
  },
});
