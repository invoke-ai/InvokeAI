import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
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
  flexShrink: 0,
  h: 'full',
  aspectRatio: '1/1',
  borderWidth: 2,
  borderRadius: 'base',
  bg: 'base.900',
  overflow: 'hidden',
  '&[data-selected="true"]': {
    borderColor: 'invokeBlue.300',
  },
} satisfies SystemStyleObject;

type Props = {
  item: S['SessionQueueItem'];
  index: number;
  isSelected: boolean;
};

export const QueueItemPreviewMini = memo(({ item, isSelected, index }: Props) => {
  const dispatch = useAppDispatch();
  const ctx = useCanvasSessionContext();
  const { imageLoaded } = useProgressData(ctx.$progressData, item.item_id);
  const imageDTO = useOutputImageDTO(item);
  const autoSwitch = useAppSelector(selectStagingAreaAutoSwitch);

  const onClick = useCallback(() => {
    ctx.$selectedItemId.set(item.item_id);
  }, [ctx.$selectedItemId, item.item_id]);

  const onDoubleClick = useCallback(() => {
    if (autoSwitch !== 'off') {
      dispatch(settingsStagingAreaAutoSwitchChanged('off'));
      toast({
        title: 'Auto-Switch Disabled',
      });
    }
  }, [autoSwitch, dispatch]);

  const onLoad = useCallback(() => {
    ctx.onImageLoad(item.item_id);
  }, [ctx, item.item_id]);

  return (
    <Flex
      id={getQueueItemElementId(index)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel item={item} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} onLoad={onLoad} asThumbnail position="absolute" />}
      {!imageLoaded && <QueueItemProgressImage itemId={item.item_id} position="absolute" />}
      <QueueItemNumber number={index + 1} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress itemId={item.item_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
