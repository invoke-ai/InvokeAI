import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { QueueItemCircularProgress } from 'features/controlLayers/components/SimpleSession/QueueItemCircularProgress';
import { QueueItemNumber } from 'features/controlLayers/components/SimpleSession/QueueItemNumber';
import { QueueItemProgressImage } from 'features/controlLayers/components/SimpleSession/QueueItemProgressImage';
import { QueueItemStatusLabel } from 'features/controlLayers/components/SimpleSession/QueueItemStatusLabel';
import { getQueueItemElementId, useOutputImageDTO } from 'features/controlLayers/components/SimpleSession/shared';
import { DndImage } from 'features/dnd/DndImage';
import { memo, useCallback, useState } from 'react';
import type { S } from 'services/api/types';

const sx = {
  cursor: 'pointer',
  userSelect: 'none',
  pos: 'relative',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
  h: 'full',
  maxH: 'full',
  maxW: 'full',
  minW: 0,
  minH: 0,
  borderWidth: 1,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    borderColor: 'invokeBlue.300',
  },
  aspectRatio: '1/1',
  flexShrink: 0,
} satisfies SystemStyleObject;

type Props = {
  item: S['SessionQueueItem'];
  number: number;
  isSelected: boolean;
  onSelectItemId: (item_id: number) => void;
  onChangeAutoSwitch: (autoSwitch: boolean) => void;
};

export const QueueItemPreviewMini = memo(({ item, isSelected, number, onSelectItemId, onChangeAutoSwitch }: Props) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const imageDTO = useOutputImageDTO(item);

  const onClick = useCallback(() => {
    onSelectItemId(item.item_id);
  }, [item.item_id, onSelectItemId]);

  const onDoubleClick = useCallback(() => {
    onChangeAutoSwitch(item.status === 'in_progress');
  }, [item.status, onChangeAutoSwitch]);

  const onLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  return (
    <Flex
      id={getQueueItemElementId(item.item_id)}
      sx={sx}
      data-selected={isSelected}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <QueueItemStatusLabel status={item.status} position="absolute" margin="auto" />
      {imageDTO && <DndImage imageDTO={imageDTO} asThumbnail onLoad={onLoad} />}
      {!imageLoaded && <QueueItemProgressImage session_id={item.session_id} position="absolute" />}
      <QueueItemNumber number={number} position="absolute" top={0} left={1} />
      <QueueItemCircularProgress
        session_id={item.session_id}
        status={item.status}
        position="absolute"
        top={1}
        right={2}
      />
    </Flex>
  );
});
QueueItemPreviewMini.displayName = 'QueueItemPreviewMini';
