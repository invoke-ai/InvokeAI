/* eslint-disable i18next/no-literal-string */
import { Divider, Flex, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { QueueItemPreviewFull } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewFull';
import { QueueItemPreviewMini } from 'features/controlLayers/components/SimpleSession/QueueItemPreviewMini';
import { memo } from 'react';
import type { S } from 'services/api/types';

export const StagingAreaContent = memo(
  ({
    items,
    selectedItem,
    selectedItemId,
    selectedItemIndex,
    onChangeAutoSwitch,
    onSelectItemId,
  }: {
    items: S['SessionQueueItem'][];
    selectedItem: S['SessionQueueItem'] | null;
    selectedItemId: number | null;
    selectedItemIndex: number | null;
    onChangeAutoSwitch: (autoSwitch: boolean) => void;
    onSelectItemId: (itemId: number) => void;
  }) => {
    return (
      <>
        <Flex position="relative" w="full" h="full" maxH="full" alignItems="center" justifyContent="center" minH={0}>
          {selectedItem && selectedItemIndex !== null && (
            <QueueItemPreviewFull
              key={`${selectedItem.item_id}-full`}
              item={selectedItem}
              number={selectedItemIndex + 1}
            />
          )}
          {!selectedItem && <Text>No generation selected</Text>}
        </Flex>
        <Divider />
        <Flex position="relative" maxW="full" w="full" h={108}>
          <ScrollableContent overflowX="scroll" overflowY="hidden">
            <Flex gap={2} w="full" h="full">
              {items.map((item, i) => (
                <QueueItemPreviewMini
                  key={`${item.item_id}-mini`}
                  item={item}
                  number={i + 1}
                  isSelected={selectedItemId === item.item_id}
                  onSelectItemId={onSelectItemId}
                  onChangeAutoSwitch={onChangeAutoSwitch}
                />
              ))}
            </Flex>
          </ScrollableContent>
        </Flex>
      </>
    );
  }
);
StagingAreaContent.displayName = 'StagingAreaContent';
