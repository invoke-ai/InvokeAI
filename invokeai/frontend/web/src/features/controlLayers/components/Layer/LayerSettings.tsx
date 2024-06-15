import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import type { LayerImageDropData } from 'features/dnd/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const LayerSettings = memo(({ id }: Props) => {
  const droppableData = useMemo<LayerImageDropData>(
    () => ({ id, actionType: 'ADD_LAYER_IMAGE', context: { id } }),
    [id]
  );

  return (
    <CanvasEntitySettings>
      PLACEHOLDER
      <IAIDroppable data={droppableData} />
    </CanvasEntitySettings>
  );
});

LayerSettings.displayName = 'LayerSettings';
