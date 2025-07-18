import { Box, Flex, forwardRef } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { QueueItemPreviewMini } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewMini';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { CSSProperties, RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { Components, ComputeItemKey, ItemContent, ListRange, VirtuosoHandle, VirtuosoProps } from 'react-virtuoso';
import { Virtuoso } from 'react-virtuoso';
import type { S } from 'services/api/types';

import { useStagingAreaContext } from './context';
import { getQueueItemElementId } from './shared';

const log = logger('system');

const virtuosoStyles = {
  width: '100%',
  height: '72px',
} satisfies CSSProperties;

/**
 * Scroll the item at the given index into view if it is not currently visible.
 */
const scrollIntoView = (
  targetIndex: number,
  rootEl: HTMLDivElement,
  virtuosoHandle: VirtuosoHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    // No range is rendered; no need to scroll to anything.
    return;
  }

  const targetItem = rootEl.querySelector(`#${getQueueItemElementId(targetIndex)}`);

  if (!targetItem) {
    if (targetIndex > range.endIndex) {
      virtuosoHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      });
    } else if (targetIndex < range.startIndex) {
      virtuosoHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      });
    } else {
      log.debug(
        `Unable to find queue item at index ${targetIndex} but it is in the rendered range ${range.startIndex}-${range.endIndex}`
      );
    }
    return;
  }

  // We found the image in the DOM, but it might be in the overscan range - rendered but not in the visible viewport.
  // Check if it is in the viewport and scroll if necessary.

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.left < rootRect.left) {
    virtuosoHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.right > rootRect.right) {
    virtuosoHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'end',
    });
  } else {
    // Image is already in view
  }

  return;
};

const useScrollableStagingArea = (rootRef: RefObject<HTMLDivElement>) => {
  const [scroller, scrollerRef] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    events: {
      initialized(osInstance) {
        // force overflow styles
        const { viewport } = osInstance.elements();
        viewport.style.overflowX = `var(--os-viewport-overflow-x)`;
        viewport.style.overflowY = `var(--os-viewport-overflow-y)`;
        viewport.style.textAlign = 'center';
      },
    },
    options: {
      scrollbars: {
        visibility: 'auto',
        autoHide: 'scroll',
        autoHideDelay: 1300,
        theme: 'os-theme-dark',
      },
      overflow: {
        y: 'hidden',
        x: 'scroll',
      },
    },
  });

  useEffect(() => {
    const { current: root } = rootRef;

    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }

    return () => {
      osInstance()?.destroy();
    };
  }, [scroller, initialize, osInstance, rootRef]);

  return scrollerRef;
};

export const StagingAreaItemsList = memo(() => {
  const canvasManager = useCanvasManager();

  const ctx = useStagingAreaContext();
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);

  const items = useStore(ctx.$items);

  const scrollerRef = useScrollableStagingArea(rootRef);

  useEffect(() => {
    return canvasManager.stagingArea.connectToSession(ctx.$items, ctx.$selectedItem);
  }, [canvasManager, ctx.$progressData, ctx.$items, ctx.$selectedItem]);

  useEffect(() => {
    return ctx.$selectedItemIndex.listen((selectedItemIndex) => {
      if (selectedItemIndex === null) {
        return;
      }
      if (!virtuosoRef.current) {
        return;
      }

      if (!rootRef.current) {
        return;
      }

      scrollIntoView(selectedItemIndex, rootRef.current, virtuosoRef.current, rangeRef.current);
    });
  }, [ctx.$selectedItemIndex]);

  const onRangeChanged = useCallback((range: ListRange) => {
    rangeRef.current = range;
  }, []);

  return (
    <Box data-overlayscrollbars-initialize="" ref={rootRef} position="relative" w="full" h="full">
      <Virtuoso<S['SessionQueueItem']>
        ref={virtuosoRef}
        data={items}
        horizontalDirection
        style={virtuosoStyles}
        computeItemKey={computeItemKey}
        increaseViewportBy={2048}
        itemContent={itemContent}
        components={components}
        rangeChanged={onRangeChanged}
        // Virtuoso expects the ref to be of HTMLElement | null | Window, but overlayscrollbars doesn't allow Window
        scrollerRef={scrollerRef as VirtuosoProps<S['SessionQueueItem'], void>['scrollerRef']}
      />
    </Box>
  );
});
StagingAreaItemsList.displayName = 'StagingAreaItemsList';

const computeItemKey: ComputeItemKey<S['SessionQueueItem'], void> = (_, item: S['SessionQueueItem']) => {
  return item.item_id;
};

const itemContent: ItemContent<S['SessionQueueItem'], void> = (index, item) => (
  <QueueItemPreviewMini key={`${item.item_id}-mini`} item={item} index={index} />
);

const listSx = {
  '& > * + *': {
    pl: 2,
  },
  '&[data-disabled="true"]': {
    filter: 'grayscale(1) opacity(0.5)',
  },
};

const components: Components<S['SessionQueueItem']> = {
  List: forwardRef(({ context: _, ...rest }, ref) => {
    const canvasManager = useCanvasManager();
    const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

    return <Flex ref={ref} sx={listSx} data-disabled={!shouldShowStagedImage} {...rest} />;
  }),
};
