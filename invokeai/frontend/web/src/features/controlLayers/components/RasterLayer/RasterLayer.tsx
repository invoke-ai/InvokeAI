import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { RasterLayerAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { Dnd } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const RasterLayer = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'raster_layer'>>(() => ({ id, type: 'raster_layer' }), [id]);
  const targetData = useMemo<Dnd.types['TargetDataTypeMap']['replaceLayerWithImage']>(
    () => Dnd.Target.replaceLayerWithImage.getData({ entityIdentifier }, entityIdentifier.id),
    [entityIdentifier]
  );

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <RasterLayerAdapterGate>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityPreviewImage />
            <CanvasEntityEditableTitle />
            <Spacer />
            <CanvasEntityHeaderCommonActions />
          </CanvasEntityHeader>
          <DndDropTarget targetData={targetData} label={t('controlLayers.replaceLayer')} />
        </CanvasEntityContainer>
      </RasterLayerAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

RasterLayer.displayName = 'RasterLayer';
