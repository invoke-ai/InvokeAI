import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { RasterLayerAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import type { ReplaceCanvasEntityObjectsWithImageActionData} from 'features/imageActions/actions';
import {replaceCanvasEntityObjectsWithImageActionApi } from 'features/imageActions/actions';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const RasterLayer = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'raster_layer'>>(() => ({ id, type: 'raster_layer' }), [id]);
  const targetData = useMemo<ReplaceCanvasEntityObjectsWithImageActionData>(
    () => replaceCanvasEntityObjectsWithImageActionApi.getData({ entityIdentifier }, entityIdentifier.id),
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
          <DndDropTarget targetData={targetData} label={t('controlLayers.replaceLayer')} isDisabled={isBusy} />
        </CanvasEntityContainer>
      </RasterLayerAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

RasterLayer.displayName = 'RasterLayer';
