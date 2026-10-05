import type { ComponentProps } from 'react';

import { Group as ChakraGroup } from '@chakra-ui/react';
import { useMemo } from 'react';

type GroupProps = ComponentProps<typeof ChakraGroup>;

/**
 * Use positional selectors because composite children may drop Chakra's data-first/between/last props. Match
 * upstream declarations and priority.
 */
const attachedCss = {
  horizontal: {
    '& > * + *': {
      borderEndStartRadius: '0 !important',
      borderStartStartRadius: '0 !important',
    },
    '& > *:not(:last-child)': {
      borderEndEndRadius: '0 !important',
      borderStartEndRadius: '0 !important',
      marginEnd: '-1px',
    },
  },
  vertical: {
    '& > * + *': {
      borderStartEndRadius: '0 !important',
      borderStartStartRadius: '0 !important',
    },
    '& > *:not(:last-child)': {
      borderEndEndRadius: '0 !important',
      borderEndStartRadius: '0 !important',
      marginBottom: '-1px',
    },
  },
} as const;

export const Group = ({ attached, css, orientation, ...props }: GroupProps) => {
  const mergedCss = useMemo(
    () => (attached ? [attachedCss[orientation === 'vertical' ? 'vertical' : 'horizontal'], css] : css),
    [attached, css, orientation]
  );

  return <ChakraGroup attached={attached} css={mergedCss} orientation={orientation} {...props} />;
};
