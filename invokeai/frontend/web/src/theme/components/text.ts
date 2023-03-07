import { defineStyle, defineStyleConfig } from '@chakra-ui/react';

const subtext = defineStyle((_props) => ({
  color: 'base.400',
}));

export const textTheme = defineStyleConfig({
  variants: {
    subtext,
  },
});
