import type { SVGProps } from 'react';

import { Box } from '@chakra-ui/react';

/** The stylized Invoke "I" logomark path, on a 44×44 viewBox. */
const INVOKE_MARK_PATH = 'M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049';

/** A currentColor/1em glyph; decorative by default, so the containing control needs an accessible name. */
export const InvokeMarkIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg
    aria-hidden="true"
    fill="none"
    height="1em"
    stroke="currentColor"
    strokeWidth="4"
    viewBox="0 0 44 44"
    width="1em"
    xmlns="http://www.w3.org/2000/svg"
    {...props}
  >
    <path d={INVOKE_MARK_PATH} />
  </svg>
);

/** The Invoke logomark, drawn in the theme brand color. */
export const InvokeMark = ({ size = 36 }: { size?: number }) => (
  <Box color="brand.fg">
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 44 44" width={size}>
      <path d={INVOKE_MARK_PATH} stroke="currentColor" strokeWidth="2.8" />
    </svg>
  </Box>
);
