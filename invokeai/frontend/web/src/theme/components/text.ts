import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const subtext = defineStyle((props) => ({
  color: mode('colors.base.500', 'colors.base.400')(props),
}));

const destructive = defineStyle((props) => ({
  color: mode('colors.error.600', 'colors.error.300')(props),
}));

export const textTheme = defineStyleConfig({
  variants: {
    subtext,
    destructive,
  },
});
