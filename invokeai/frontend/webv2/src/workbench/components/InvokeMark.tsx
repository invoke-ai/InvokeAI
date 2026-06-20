import { Box } from '@chakra-ui/react';

/** The Invoke logomark, drawn in the theme brand color. */
export const InvokeMark = ({ size = 36 }: { size?: number }) => (
  <Box color="brand.fg">
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 44 44" width={size}>
      <path
        d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
        stroke="currentColor"
        strokeWidth="2.8"
      />
    </svg>
  </Box>
);
