import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
import { memo, useCallback, useMemo } from 'react';
import type { S } from 'services/api/types';

import { useOutputImageDTO, useStagingAreaContext } from './context2';

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
};

export const QueueItemPreviewMini = memo(({ item, index }: Props) => {
  const ctx = useStagingAreaContext();
  const dispatch = useAppDispatch();
  const $isSelected = useMemo(() => ctx.buildIsSelectedComputed(item.item_id), [ctx, item.item_id]);
  const isSelected = useStore($isSelected);
  const imageDTO = useOutputImageDTO(item.item_id);
  const autoSwitch = useAppSelector(selectStagingAreaAutoSwitch);

  const onClick = useCallback(() => {
    ctx.select(item.item_id);
  }, [ctx, item.item_id]);

  const onDoubleClick = useCallback(() => {
    if (autoSwitch !== 'off') {
      dispatch(settingsStagingAreaAutoSwitchChanged('off'));
      toast({
        title: 'Auto-Switch Disabled',
      });
    }
  }, [autoSwitch, dispatch]);

  return (
    <Flex
      id={getQueueItemElementId(index)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel item={item} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} asThumbnail position="absolute" />}
      <QueueItemProgressImage itemId={item.item_id} position="absolute" />
      <QueueItemNumber number={index + 1} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress itemId={item.item_id} status={item.status} position="absolute" top={1} right={2} />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
