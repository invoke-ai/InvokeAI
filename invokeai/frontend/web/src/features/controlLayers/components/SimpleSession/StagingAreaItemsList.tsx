/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemPreviewMini } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewMini';
import { memo } from 'react';

export const StagingAreaItemsList = memo(() => {
  const ctx = useCanvasSessionContext();
  const items = useStore(ctx.$items);
  const selectedItemId = useStore(ctx.$selectedItemId);

  return (
    <ScrollableContent overflowX="scroll" overflowY="hidden">
      <Flex gap={2} w="full" h="full">
        {items.map((item, i) => (
          <QueueItemPreviewMini
            key={`${item.item_id}-mini`}
            item={item}
            number={i + 1}
            isSelected={selectedItemId === item.item_id}
            onSelectItemId={ctx.$selectedItemId.set}
            onChangeAutoSwitch={ctx.$autoSwitch.set}
          />
        ))}
      </Flex>
    </ScrollableContent>
  );
});
StagingAreaItemsList.displayName = 'StagingAreaItemsList';
