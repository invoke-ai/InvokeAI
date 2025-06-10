/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemPreviewMini } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewMini';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useEffect } from 'react';

export const StagingAreaItemsList = memo(() => {
  const canvasManager = useCanvasManagerSafe();
  const ctx = useCanvasSessionContext();
  const items = useStore(ctx.$items);
  const selectedItemId = useStore(ctx.$selectedItemId);

  useEffect(() => {
    if (!canvasManager) {
      return;
    }

    return canvasManager.stagingArea.connectToSession(ctx.$selectedItemId, ctx.$progressData);
  }, [canvasManager, ctx.$progressData, ctx.$selectedItemId]);

  return (
    <ScrollableContent overflowX="scroll" overflowY="hidden">
      <Flex gap={2} w="full" h="full" justifyContent="safe center">
        {items.map((item, i) => (
          <QueueItemPreviewMini
            key={`${item.item_id}-mini`}
            item={item}
            number={i + 1}
            isSelected={selectedItemId === item.item_id}
          />
        ))}
      </Flex>
    </ScrollableContent>
  );
});
StagingAreaItemsList.displayName = 'StagingAreaItemsList';
