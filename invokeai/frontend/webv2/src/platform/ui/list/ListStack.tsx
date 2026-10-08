import type { ReactNode } from 'react';

import { Stack, type StackProps } from '@chakra-ui/react';
import { Children } from 'react';

import { IN_FLOW_DIVIDER_HIDING_CSS, ListDivider } from './ListDivider';
import { LIST_ROW_GAP_PX } from './listLayout';

export interface ListStackProps extends Omit<StackProps, 'children' | 'gap' | 'role'> {
  /** Accessible name of the list. */
  label: string;
  /** Hairlines between rows, inset from the edges. */
  dividers?: boolean;
  /** Rows, each rendering `role="listitem"` (ListItem does). */
  children: ReactNode;
}

/**
 * A short list rendered in full, for rows that are few or bounded; long or growing lists use `List`, which
 * virtualizes. Rows keep the family's spacing either way.
 */
export const ListStack = ({ children, dividers = false, label, ...stackProps }: ListStackProps) => (
  <Stack
    aria-label={label}
    css={dividers ? IN_FLOW_DIVIDER_HIDING_CSS : undefined}
    gap={dividers ? 0 : `${LIST_ROW_GAP_PX}px`}
    role="list"
    {...stackProps}
  >
    {dividers
      ? Children.toArray(children).flatMap((row, index) =>
          // Dividers carry no state, so index keys are safe for them.
          index === 0 ? [row] : [<ListDivider key={`divider-${index}`} placement="in-flow" />, row]
        )
      : children}
  </Stack>
);
