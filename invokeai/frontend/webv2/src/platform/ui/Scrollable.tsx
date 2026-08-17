import type { ComponentProps, ReactNode, Ref } from 'react';

import { ScrollArea } from '@chakra-ui/react';

type ScrollAreaRootProps = ComponentProps<typeof ScrollArea.Root>;
type ScrollAreaContentProps = ComponentProps<typeof ScrollArea.Content>;

/**
 * The workbench's standard scroll container: ScrollArea with hover-revealed
 * scrollbars and the content wrapper zag needs for correct thumb sizing.
 * Layout props (h, maxH, flex, ...) go to the root.
 */
export const Scrollable = ({
  children,
  contentProps,
  label,
  orientation = 'vertical',
  viewportRef,
  ...rootProps
}: ScrollAreaRootProps & {
  children: ReactNode;
  /** Extra props for the content wrapper, e.g. to let children fill the viewport height. */
  contentProps?: ScrollAreaContentProps;
  /** Accessible name for the scroll viewport. */
  label?: string;
  /** Scroll axis; the scrollbar and content sizing follow it. Defaults to vertical. */
  orientation?: 'horizontal' | 'vertical';
  /** The scrolling element itself — what a virtualizer needs to observe. */
  viewportRef?: Ref<HTMLDivElement>;
}) => (
  <ScrollArea.Root size="xs" variant="hover" {...rootProps}>
    <ScrollArea.Viewport aria-label={label} h="full" ref={viewportRef} role={label ? 'region' : undefined} w="full">
      <ScrollArea.Content w={orientation === 'horizontal' ? 'max-content' : 'full'} {...contentProps}>
        {children}
      </ScrollArea.Content>
    </ScrollArea.Viewport>
    <ScrollArea.Scrollbar orientation={orientation}>
      <ScrollArea.Thumb />
    </ScrollArea.Scrollbar>
  </ScrollArea.Root>
);
