import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { ControlLayerBadges } from 'features/controlLayers/components/ControlLayer/ControlLayerBadges';
import { ControlLayerSettings } from 'features/controlLayers/components/ControlLayer/ControlLayerSettings';
import { CanvasEntityStateGate } from 'features/controlLayers/contexts/CanvasEntityStateGate';
import { ControlLayerAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReplaceCanvasEntityObjectsWithImageDndTargetData } from 'features/dnd/dnd';
import { replaceCanvasEntityObjectsWithImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const ControlLayer = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'control_layer'>>(
    () => ({ id, type: 'control_layer' }),
    [id]
  );
  const dndTargetData = useMemo<ReplaceCanvasEntityObjectsWithImageDndTargetData>(
    () => replaceCanvasEntityObjectsWithImageDndTarget.getData({ entityIdentifier }, entityIdentifier.id),
    [entityIdentifier]
  );

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <ControlLayerAdapterGate>
        <CanvasEntityStateGate entityIdentifier={entityIdentifier}>
          <CanvasEntityContainer>
            <CanvasEntityHeader>
              <CanvasEntityPreviewImage />
              <CanvasEntityEditableTitle />
              <Spacer />
              <ControlLayerBadges />
              <CanvasEntityHeaderCommonActions />
            </CanvasEntityHeader>
            <CanvasEntitySettingsWrapper>
              <ControlLayerSettings />
            </CanvasEntitySettingsWrapper>
            <DndDropTarget
              dndTarget={replaceCanvasEntityObjectsWithImageDndTarget}
              dndTargetData={dndTargetData}
              label={t('controlLayers.replaceLayer')}
              isDisabled={isBusy}
            />
          </CanvasEntityContainer>
        </CanvasEntityStateGate>
      </ControlLayerAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

ControlLayer.displayName = 'ControlLayer';
