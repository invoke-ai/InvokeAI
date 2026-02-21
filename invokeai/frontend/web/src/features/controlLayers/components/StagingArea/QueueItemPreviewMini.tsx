import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { QueueItemCircularProgress } from 'features/controlLayers/components/StagingArea/QueueItemCircularProgress';
import { QueueItemProgressImage } from 'features/controlLayers/components/StagingArea/QueueItemProgressImage';
import { QueueItemStatusLabel } from 'features/controlLayers/components/StagingArea/QueueItemStatusLabel';
import { getQueueItemElementId } from 'features/controlLayers/components/StagingArea/shared';
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { DndImage } from 'features/dnd/DndImage';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';

import { useStagingAreaContext } from './context';
import { QueueItemNumber } from './QueueItemNumber';
import type { StagingEntry } from './state';

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
  entry: StagingEntry;
  index: number;
};

export const QueueItemPreviewMini = memo(({ entry, index }: Props) => {
  const ctx = useStagingAreaContext();
  const dispatch = useAppDispatch();
  const $isSelected = useMemo(
    () => ctx.buildIsSelectedComputed(entry.item.item_id, entry.imageIndex),
    [ctx, entry.item.item_id, entry.imageIndex]
  );
  const isSelected = useStore($isSelected);
  const autoSwitch = useAppSelector(selectStagingAreaAutoSwitch);

  const onClick = useCallback(() => {
    ctx.select(entry.item.item_id, entry.imageIndex);
  }, [ctx, entry.item.item_id, entry.imageIndex]);

  const onDoubleClick = useCallback(() => {
    if (autoSwitch !== 'off') {
      dispatch(settingsStagingAreaAutoSwitchChanged('off'));
      toast({
        title: 'Auto-Switch Disabled',
      });
    }
  }, [autoSwitch, dispatch]);

  const onLoad = useCallback(() => {
    ctx.onImageLoaded(entry.item.item_id);
  }, [ctx, entry.item.item_id]);

  return (
    <Flex
      id={getQueueItemElementId(index)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel item={entry.item} position="absolute" margin="auto" />
      {entry.imageDTO && <DndImage imageDTO={entry.imageDTO} position="absolute" onLoad={onLoad} />}
      <QueueItemProgressImage itemId={entry.item.item_id} position="absolute" />
      <QueueItemNumber number={index + 1} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress
        itemId={entry.item.item_id}
        status={entry.item.status}
        position="absolute"
        top={1}
        right={2}
      />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
