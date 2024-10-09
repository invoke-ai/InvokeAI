import { Spacer } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { ControlLayerBadges } from 'features/controlLayers/components/ControlLayer/ControlLayerBadges';
import { ControlLayerControlAdapter } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapter';
import { ControlLayerAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReplaceLayerImageDropData } from 'features/dnd/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const ControlLayer = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'control_layer'>>(
    () => ({ id, type: 'control_layer' }),
    [id]
  );
  const dropData = useMemo<ReplaceLayerImageDropData>(
    () => ({ id, actionType: 'REPLACE_LAYER_WITH_IMAGE', context: { entityIdentifier } }),
    [id, entityIdentifier]
  );
  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <ControlLayerAdapterGate>
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
          <IAIDroppable data={dropData} dropLabel={t('controlLayers.replaceLayer')} />
        </CanvasEntityContainer>
      </ControlLayerAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

ControlLayer.displayName = 'ControlLayer';
