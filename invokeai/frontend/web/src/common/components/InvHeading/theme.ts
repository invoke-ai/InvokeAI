import { defineStyle, defineStyleConfig } from '@chakra-ui/react';

const blue = defineStyle(() => ({
  color: 'blue.300',
}));

export const headingTheme = defineStyleConfig({
  variants: {
    blue,
  },
});
