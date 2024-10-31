import { Grid, GridItem } from '@invoke-ai/ui-library';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { Dnd } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const addRasterLayerFromImageDndTargetData = Dnd.Target.newRasterLayerFromImage.getData();
const addControlLayerFromImageDndTargetData = Dnd.Target.newControlLayerFromImage.getData();
const addRegionalGuidanceReferenceImageFromImageDndTargetData =
  Dnd.Target.newRegionalGuidanceReferenceImageFromImage.getData();
const addGlobalReferenceImageFromImageDndTargetData = Dnd.Target.newGlobalReferenceImageFromImage.getData();

export const CanvasDropArea = memo(() => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();

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
          <DndDropTarget
            targetData={addRasterLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRasterLayer')}
            isDisabled={isBusy}
          />
        </GridItem>
        <GridItem position="relative">
          <DndDropTarget
            targetData={addControlLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newControlLayer')}
            isDisabled={isBusy}
          />
        </GridItem>

        <GridItem position="relative">
          <DndDropTarget
            targetData={addRegionalGuidanceReferenceImageFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRegionalReferenceImage')}
            isDisabled={isBusy}
          />
        </GridItem>
        <GridItem position="relative">
          <DndDropTarget
            targetData={addGlobalReferenceImageFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newGlobalReferenceImage')}
            isDisabled={isBusy}
          />
        </GridItem>
      </Grid>
    </>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
