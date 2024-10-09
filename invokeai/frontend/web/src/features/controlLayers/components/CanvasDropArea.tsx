import { Grid, GridItem } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import type {
  AddControlLayerFromImageDropData,
  AddGlobalReferenceImageFromImageDropData,
  AddRasterLayerFromImageDropData,
  AddRegionalReferenceImageFromImageDropData,
} from 'features/dnd/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const addRasterLayerFromImageDropData: AddRasterLayerFromImageDropData = {
  id: 'add-raster-layer-from-image-drop-data',
  actionType: 'ADD_RASTER_LAYER_FROM_IMAGE',
};

const addControlLayerFromImageDropData: AddControlLayerFromImageDropData = {
  id: 'add-control-layer-from-image-drop-data',
  actionType: 'ADD_CONTROL_LAYER_FROM_IMAGE',
};

const addRegionalReferenceImageFromImageDropData: AddRegionalReferenceImageFromImageDropData = {
  id: 'add-control-layer-from-image-drop-data',
  actionType: 'ADD_REGIONAL_REFERENCE_IMAGE_FROM_IMAGE',
};

const addGlobalReferenceImageFromImageDropData: AddGlobalReferenceImageFromImageDropData = {
  id: 'add-control-layer-from-image-drop-data',
  actionType: 'ADD_GLOBAL_REFERENCE_IMAGE_FROM_IMAGE',
};

export const CanvasDropArea = memo(() => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();

  if (imageViewer.isOpen) {
    return null;
  }

  return (
    <>
      <Grid
        gridTemplateRows="1fr 1fr"
        gridTemplateColumns="1fr 1fr"
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        pointerEvents="none"
      >
        <GridItem position="relative">
          <IAIDroppable
            dropLabel={t('controlLayers.canvasContextMenu.newRasterLayer')}
            data={addRasterLayerFromImageDropData}
          />
        </GridItem>
        <GridItem position="relative">
          <IAIDroppable
            dropLabel={t('controlLayers.canvasContextMenu.newControlLayer')}
            data={addControlLayerFromImageDropData}
          />
        </GridItem>
        <GridItem position="relative">
          <IAIDroppable
            dropLabel={t('controlLayers.canvasContextMenu.newRegionalReferenceImage')}
            data={addRegionalReferenceImageFromImageDropData}
          />
        </GridItem>
        <GridItem position="relative">
          <IAIDroppable
            dropLabel={t('controlLayers.canvasContextMenu.newGlobalReferenceImage')}
            data={addGlobalReferenceImageFromImageDropData}
          />
        </GridItem>
      </Grid>
    </>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
