import { Grid, GridItem } from '@invoke-ai/ui-library';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { newCanvasEntityFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const addRasterLayerFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({ type: 'raster_layer' });
const addControlLayerFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({
  type: 'control_layer',
});
const addRegionalGuidanceReferenceImageFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({
  type: 'regional_guidance_with_reference_image',
});
const addResizedControlLayerFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({
  type: 'control_layer',
  withResize: true,
});

export const CanvasDropArea = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();

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
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addRasterLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRasterLayer')}
            isDisabled={isBusy}
          />
        </GridItem>
        <GridItem position="relative">
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addControlLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newControlLayer')}
            isDisabled={isBusy}
          />
        </GridItem>
        <GridItem position="relative">
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addRegionalGuidanceReferenceImageFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRegionalReferenceImage')}
            isDisabled={isBusy}
          />
        </GridItem>
        <GridItem position="relative">
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addResizedControlLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newResizedControlLayer')}
            isDisabled={isBusy}
          />
        </GridItem>
      </Grid>
    </>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
