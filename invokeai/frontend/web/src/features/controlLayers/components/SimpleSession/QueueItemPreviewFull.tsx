import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import {
  useCanvasSessionContext,
  useOutputImageDTO,
  useProgressData,
} from 'features/controlLayers/components/SimpleSession/context';
import { ImageActions } from 'features/controlLayers/components/SimpleSession/ImageActions';
import { QueueItemCircularProgress } from 'features/controlLayers/components/SimpleSession/QueueItemCircularProgress';
import { QueueItemNumber } from 'features/controlLayers/components/SimpleSession/QueueItemNumber';
import { QueueItemProgressImage } from 'features/controlLayers/components/SimpleSession/QueueItemProgressImage';
import { QueueItemStatusLabel } from 'features/controlLayers/components/SimpleSession/QueueItemStatusLabel';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { DndImage } from 'features/dnd/DndImage';
import { memo } from 'react';
import type { S } from 'services/api/types';

type Props = {
  item: S['SessionQueueItem'];
  number: number;
};

const sx = {
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
  h: 'full',
  w: 'full',
} satisfies SystemStyleObject;

export const QueueItemPreviewFull = memo(({ item, number }: Props) => {
  const ctx = useCanvasSessionContext();
  const imageDTO = useOutputImageDTO(item);
  const { imageLoaded } = useProgressData(ctx.$progressData, item.item_id);

  return (
    <Flex id={getQueueItemElementId(item.item_id)} sx={sx}>
      <QueueItemStatusLabel item={item} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} />}
      {!imageLoaded && <QueueItemProgressImage itemId={item.item_id} position="absolute" />}
      {imageDTO && <ImageActions imageDTO={imageDTO} position="absolute" top={1} right={2} />}
      <QueueItemNumber number={number} position="absolute" top={1} left={2} />
      <QueueItemCircularProgress
        itemId={item.item_id}
        status={item.status}
        position="absolute"
        top={1}
        right={2}
        size={8}
      />
    </Flex>
  );
});
QueueItemPreviewFull.displayName = 'QueueItemPreviewFull';
