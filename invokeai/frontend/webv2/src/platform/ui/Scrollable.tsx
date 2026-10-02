import type { ComponentProps, ReactNode, Ref } from 'react';

import { mergeRefs, ScrollArea } from '@chakra-ui/react';
import { usePreservedScrollOffset } from '@platform/react/usePreservedScrollOffset';
import { useCallback, useRef } from 'react';

import { useScrollAreaPhantomHeal } from './useScrollAreaPhantomHeal';

type ScrollAreaRootProps = ComponentProps<typeof ScrollArea.Root>;
type ScrollAreaContentProps = ComponentProps<typeof ScrollArea.Content>;
type ScrollAreaViewportProps = ComponentProps<typeof ScrollArea.Viewport>;

/**
 * Override Zag's inline min-width in vertical areas; long unbroken content would otherwise create unreachable
 * horizontal overflow.
 */
const VERTICAL_CONTENT_STYLE = { minWidth: 0 } as const;

/** Layout props target the root; the content wrapper is required for Zag's thumb sizing. */
export const Scrollable = ({
  children,
  contentProps,
  label,
  orientation = 'vertical',
  viewportProps,
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
  /** Extra props for the scrolling viewport itself, e.g. scroll/focus handlers. */
  viewportProps?: ScrollAreaViewportProps;
  /** Ref to the scrolling viewport; callback refs support mount-time offset restoration. */
  viewportRef?: Ref<HTMLDivElement | null>;
}) => {
  const fallbackViewportRef = useRef<HTMLDivElement | null>(null);
  const resolvedViewportRef = viewportRef && typeof viewportRef !== 'function' ? viewportRef : fallbackViewportRef;
  const viewportCallback = typeof viewportRef === 'function' ? viewportRef : undefined;
  // Compose callback refs at attachment time, not during render.
  const attachViewport = useCallback(
    (element: HTMLDivElement | null) => mergeRefs(fallbackViewportRef, viewportCallback)(element),
    [viewportCallback]
  );

  // Hidden mounted widgets can lose scroll offsets; restore them when shown.
  usePreservedScrollOffset(resolvedViewportRef);

  useScrollAreaPhantomHeal(resolvedViewportRef);

  return (
    <ScrollArea.Root size="xs" variant="hover" {...rootProps}>
      <ScrollArea.Viewport
        aria-label={label}
        h="full"
        role={label ? 'region' : undefined}
        w="full"
        {...viewportProps}
        ref={viewportCallback ? attachViewport : resolvedViewportRef}
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
