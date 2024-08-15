import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { RasterLayerActionsMenu } from 'features/controlLayers/components/RasterLayer/RasterLayerActionsMenu';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const RasterLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'raster_layer' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <RasterLayerActionsMenu />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

RasterLayer.displayName = 'RasterLayer';
