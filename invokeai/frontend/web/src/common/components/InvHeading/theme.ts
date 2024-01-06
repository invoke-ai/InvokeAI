import { defineStyle, defineStyleConfig } from '@chakra-ui/react';

const invokeBlue = defineStyle(() => ({
  color: 'invokeBlue.300',
}));

export const headingTheme = defineStyleConfig({
  variants: {
    invokeBlue,
  },
});
