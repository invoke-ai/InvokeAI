import { useDisclosure } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { LayerHeader } from 'features/controlLayers/components/Layer/LayerHeader';
import { LayerSettings } from 'features/controlLayers/components/Layer/LayerSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { LayerImageDropData } from 'features/dnd/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const Layer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'layer' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });
  const droppableData = useMemo<LayerImageDropData>(
    () => ({ id, actionType: 'ADD_LAYER_IMAGE', context: { id } }),
    [id]
  );

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <LayerHeader onToggleVisibility={onToggle} />
        {isOpen && <LayerSettings />}
        <IAIDroppable data={droppableData} />
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

Layer.displayName = 'Layer';
