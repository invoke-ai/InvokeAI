/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemPreviewMini } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewMini';
import { getOutputImageName } from 'features/controlLayers/components/SimpleSession/shared';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { effect } from 'nanostores';
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

    return effect([ctx.$selectedItem, ctx.$progressData], (selectedItem, progressData) => {
      if (!selectedItem) {
        canvasManager.stagingArea.render();
        return;
      }

      const outputImageName = getOutputImageName(selectedItem);

      if (outputImageName) {
        canvasManager.stagingArea.render({ type: 'imageName', data: outputImageName });
        return;
      }

      const data = progressData[selectedItem.item_id];

      if (data?.progressImage) {
        canvasManager.stagingArea.render({ type: 'dataURL', data: data.progressImage.dataURL });
        return;
      }

      canvasManager.stagingArea.render();
    });
  }, [canvasManager, ctx.$progressData, ctx.$selectedItem]);

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
