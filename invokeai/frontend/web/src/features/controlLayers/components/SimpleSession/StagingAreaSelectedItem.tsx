/* eslint-disable i18next/no-literal-string */
import { Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemPreviewFull } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewFull';
import { memo } from 'react';

export const StagingAreaSelectedItem = memo(() => {
  const ctx = useCanvasSessionContext();
  const selectedItem = useStore(ctx.$selectedItem);
  const selectedItemIndex = useStore(ctx.$selectedItemIndex);

  if (selectedItem && selectedItemIndex !== null) {
    return (
      <QueueItemPreviewFull key={`${selectedItem.item_id}-full`} item={selectedItem} number={selectedItemIndex + 1} />
    );
  }

  return <Text>No generation selected</Text>;
});
StagingAreaSelectedItem.displayName = 'StagingAreaSelectedItem';
