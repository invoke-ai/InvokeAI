import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { ControlLayerControlAdapter } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapter';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { EntityLayerAdapterProviderGate } from 'features/controlLayers/hooks/useEntityLayerAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const ControlLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'control_layer' }), [id]);

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
          <CanvasEntitySettingsWrapper>
            <ControlLayerControlAdapter />
          </CanvasEntitySettingsWrapper>
        </CanvasEntityContainer>
      </EntityLayerAdapterProviderGate>
    </EntityIdentifierContext.Provider>
  );
});

ControlLayer.displayName = 'ControlLayer';
