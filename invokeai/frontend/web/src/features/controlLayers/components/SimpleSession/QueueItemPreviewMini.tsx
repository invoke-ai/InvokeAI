import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemCircularProgress } from 'features/controlLayers/components/SimpleSession/QueueItemCircularProgress';
import { QueueItemNumber } from 'features/controlLayers/components/SimpleSession/QueueItemNumber';
import { QueueItemProgressImage } from 'features/controlLayers/components/SimpleSession/QueueItemProgressImage';
import { QueueItemStatusLabel } from 'features/controlLayers/components/SimpleSession/QueueItemStatusLabel';
import { getQueueItemElementId, useOutputImageDTO } from 'features/controlLayers/components/SimpleSession/shared';
import { DndImage } from 'features/dnd/DndImage';
import { memo, useCallback, useState } from 'react';
import type { S } from 'services/api/types';

const sx = {
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  h: 108,
  w: 108,
  flexShrink: 0,
  borderWidth: 1,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    borderColor: 'invokeBlue.300',
  },
} satisfies SystemStyleObject;

type Props = {
  item: S['SessionQueueItem'];
  number: number;
  isSelected: boolean;
};

export const QueueItemPreviewMini = memo(({ item, isSelected, number }: Props) => {
  const ctx = useCanvasSessionContext();
  const [imageLoaded, setImageLoaded] = useState(false);
  const imageDTO = useOutputImageDTO(item);

  const onClick = useCallback(() => {
    ctx.$selectedItemId.set(item.item_id);
  }, [ctx.$selectedItemId, item.item_id]);

  const onDoubleClick = useCallback(() => {
    ctx.$autoSwitch.set(item.status === 'in_progress');
  }, [ctx.$autoSwitch, item.status]);

  const onLoad = useCallback(() => {
    setImageLoaded(true);
    ctx.$lastLoadedItemId.set(item.item_id);
  }, [ctx.$lastLoadedItemId, item.item_id]);

  return (
    <Flex
      id={getQueueItemElementId(item.item_id)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel status={item.status} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} onLoad={onLoad} asThumbnail />}
      {!imageLoaded && <QueueItemProgressImage itemId={item.item_id} position="absolute" />}
      <QueueItemNumber number={number} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress itemId={item.item_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
