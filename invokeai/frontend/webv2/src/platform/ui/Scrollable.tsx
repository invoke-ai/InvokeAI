import type { ComponentProps, ReactNode, RefObject } from 'react';

import { ScrollArea } from '@chakra-ui/react';
import { usePreservedScrollOffset } from '@platform/react/usePreservedScrollOffset';
import { useRef } from 'react';

type ScrollAreaRootProps = ComponentProps<typeof ScrollArea.Root>;
type ScrollAreaContentProps = ComponentProps<typeof ScrollArea.Content>;

/**
 * zag pins `min-width: fit-content` *inline* on every scroll-area content box,
 * so the box grows to its content's min-content width. A horizontal strip wants
 * exactly that. In a vertical area it is a trap: one unbreakable string (a long
 * name, an unwrapped identifier) widens the content box past the viewport, the
 * area scrolls sideways — and since a vertical area renders no horizontal
 * scrollbar, the overflow is simply unreachable. Vertical areas override it back
 * to zero, so their content stretches to the viewport and truncation inside is
 * what gives. Only an inline style can beat an inline style.
 */
const VERTICAL_CONTENT_STYLE = { minWidth: 0 } as const;

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
  viewportRef?: RefObject<HTMLDivElement | null>;
}) => {
  const fallbackViewportRef = useRef<HTMLDivElement | null>(null);
  // One ref, shared with the caller when it wants one, so nothing has to merge
  // or reassign refs during render.
  const resolvedViewportRef = viewportRef ?? fallbackViewportRef;

  // The shell keeps widgets mounted across layout switches, and a scroll
  // container that stops being rendered loses its offset outright.
  usePreservedScrollOffset(resolvedViewportRef);

  return (
    <ScrollArea.Root size="xs" variant="hover" {...rootProps}>
      <ScrollArea.Viewport
        aria-label={label}
        h="full"
        ref={resolvedViewportRef}
        role={label ? 'region' : undefined}
        w="full"
      >
        <ScrollArea.Content
          style={orientation === 'horizontal' ? undefined : VERTICAL_CONTENT_STYLE}
          w={orientation === 'horizontal' ? 'max-content' : 'full'}
          {...contentProps}
        >
          {children}
        </ScrollArea.Content>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation={orientation}>
        <ScrollArea.Thumb />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  );
};
