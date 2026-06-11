import { ScrollArea } from '@chakra-ui/react';
import type { ComponentProps, ReactNode } from 'react';

type ScrollAreaRootProps = ComponentProps<typeof ScrollArea.Root>;

/**
 * The workbench's standard scroll container: ScrollArea with hover-revealed
 * scrollbars and the content wrapper zag needs for correct thumb sizing.
 * Layout props (h, maxH, flex, ...) go to the root.
 */
export const Scrollable = ({
  children,
  label,
  ...rootProps
}: ScrollAreaRootProps & {
  children: ReactNode;
  /** Accessible name for the scroll viewport. */
  label?: string;
}) => (
  <ScrollArea.Root size="xs" variant="hover" {...rootProps}>
    <ScrollArea.Viewport aria-label={label} h="full" w="full">
      <ScrollArea.Content w="full">{children}</ScrollArea.Content>
    </ScrollArea.Viewport>
    <ScrollArea.Scrollbar>
      <ScrollArea.Thumb />
    </ScrollArea.Scrollbar>
  </ScrollArea.Root>
);
