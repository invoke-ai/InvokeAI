import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { ControlLayerBadges } from 'features/controlLayers/components/ControlLayer/ControlLayerBadges';
import { ControlLayerControlAdapter } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapter';
import { EntityLayerAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const ControlLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'control_layer' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <EntityLayerAdapterGate>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityPreviewImage />
            <CanvasEntityEditableTitle />
            <Spacer />
            <ControlLayerBadges />
            <CanvasEntityHeaderCommonActions />
          </CanvasEntityHeader>
          <CanvasEntitySettingsWrapper>
            <ControlLayerControlAdapter />
          </CanvasEntitySettingsWrapper>
        </CanvasEntityContainer>
      </EntityLayerAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

ControlLayer.displayName = 'ControlLayer';
