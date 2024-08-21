import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { EntityLayerAdapterProviderGate } from 'features/controlLayers/hooks/useEntityLayerAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const RasterLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'raster_layer' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <EntityLayerAdapterProviderGate>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityEnabledToggle />
            <CanvasEntityEditableTitle />
            <Spacer />
            <CanvasEntityDeleteButton />
          </CanvasEntityHeader>
        </CanvasEntityContainer>
      </EntityLayerAdapterProviderGate>
    </EntityIdentifierContext.Provider>
  );
});

RasterLayer.displayName = 'RasterLayer';
