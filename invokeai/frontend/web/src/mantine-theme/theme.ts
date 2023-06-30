import { MantineThemeOverride } from '@mantine/core';

export const mantineTheme: MantineThemeOverride = {
  colorScheme: 'dark',
  fontFamily: `'Inter Variable', sans-serif`,
  components: {
    ScrollArea: {
      defaultProps: {
        scrollbarSize: 10,
      },
      styles: {
        scrollbar: {
          '&:hover': {
            backgroundColor: 'var(--invokeai-colors-baseAlpha-300)',
          },
        },
        thumb: {
          backgroundColor: 'var(--invokeai-colors-baseAlpha-300)',
        },
      },
    },
  },
};
