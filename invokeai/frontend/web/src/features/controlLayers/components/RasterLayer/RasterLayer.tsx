import { Spacer } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { CanvasEntityTitleEdit } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const RasterLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'raster_layer' }), [id]);
  const editing = useBoolean(false);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={editing.setTrue}>
          <CanvasEntityEnabledToggle />
          {editing.isTrue ? <CanvasEntityTitleEdit onStopEditing={editing.setFalse} /> : <CanvasEntityTitle />}
          <Spacer />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

RasterLayer.displayName = 'RasterLayer';
