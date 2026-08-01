import type { ComponentProps } from 'react';

import { Group as ChakraGroup } from '@chakra-ui/react';
import { useMemo } from 'react';

type GroupProps = ComponentProps<typeof ChakraGroup>;

/**
 * Chakra's `Group`, with `attached` made to work structurally.
 *
 * Upstream, `attached` clones each *React* child with `data-first` /
 * `data-between` / `data-last` and styles `& > *[data-first]`. That only reaches
 * the DOM when the child forwards unknown props to its root element — so the
 * moment a group holds a component of ours rather than a bare Chakra control,
 * the attributes are dropped on the floor and the seam silently renders as a row
 * of separately-rounded buttons. Nothing errors; it just looks wrong.
 *
 * Selecting on position instead of on an attribute removes the dependency: the
 * browser knows which child is first and last whether or not React told it. The
 * declarations mirror upstream's exactly, `!important` included, so this only
 * ever adds the cases upstream misses.
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
