import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import {
  useCanvasSessionContext,
  useOutputImageDTO,
  useProgressData,
} from 'features/controlLayers/components/SimpleSession/context';
import { QueueItemCircularProgress } from 'features/controlLayers/components/SimpleSession/QueueItemCircularProgress';
import { QueueItemNumber } from 'features/controlLayers/components/SimpleSession/QueueItemNumber';
import { QueueItemProgressImage } from 'features/controlLayers/components/SimpleSession/QueueItemProgressImage';
import { QueueItemStatusLabel } from 'features/controlLayers/components/SimpleSession/QueueItemStatusLabel';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { DndImage } from 'features/dnd/DndImage';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import type { S } from 'services/api/types';

const sx = {
  cursor: 'pointer',
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  h: 108,
  w: 108,
  flexShrink: 0,
  borderWidth: 2,
  borderRadius: 'base',
  bg: 'base.900',
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
  const { imageLoaded } = useProgressData(ctx.$progressData, item.item_id);
  const imageDTO = useOutputImageDTO(item);

  const onClick = useCallback(() => {
    ctx.$selectedItemId.set(item.item_id);
  }, [ctx.$selectedItemId, item.item_id]);

  const onDoubleClick = useCallback(() => {
    const autoSwitch = ctx.$autoSwitch.get();
    if (autoSwitch !== 'off') {
      ctx.$autoSwitch.set('off');
      toast({
        title: 'Auto-Switch Disabled',
      });
    }
  }, [ctx.$autoSwitch]);

  const onLoad = useCallback(() => {
    ctx.onImageLoad(item.item_id);
  }, [ctx, item.item_id]);

  return (
    <Flex
      id={getQueueItemElementId(item.item_id)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel item={item} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} onLoad={onLoad} asThumbnail />}
      {!imageLoaded && <QueueItemProgressImage itemId={item.item_id} position="absolute" />}
      <QueueItemNumber number={number} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress itemId={item.item_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
