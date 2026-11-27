import { Grid, GridItem } from '@invoke-ai/ui-library';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useIsEntityTypeEnabled } from 'features/controlLayers/hooks/useIsEntityTypeEnabled';
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
const addInpaintMaskFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({ type: 'inpaint_mask' });
const addResizedControlLayerFromImageDndTargetData = newCanvasEntityFromImageDndTarget.getData({
  type: 'control_layer',
  withResize: true,
});

export const CanvasDropArea = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const isRasterLayerEnabled = useIsEntityTypeEnabled('raster_layer');
  const isControlLayerEnabled = useIsEntityTypeEnabled('control_layer');
  const isRegionalGuidanceEnabled = useIsEntityTypeEnabled('regional_guidance');
  const isInpaintMaskEnabled = useIsEntityTypeEnabled('inpaint_mask');

  return (
    <>
      <Grid
        gridTemplateRows="1fr 1fr"
        gridTemplateColumns="repeat(6, 1fr)"
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        pointerEvents="none"
      >
        <GridItem position="relative" colSpan={3}>
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addRasterLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRasterLayer')}
            isDisabled={isBusy || !isRasterLayerEnabled}
          />
        </GridItem>
        <GridItem position="relative" colSpan={3}>
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addControlLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newControlLayer')}
            isDisabled={isBusy || !isControlLayerEnabled}
          />
        </GridItem>
        <GridItem position="relative" colSpan={2}>
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addRegionalGuidanceReferenceImageFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newRegionalReferenceImage')}
            isDisabled={isBusy || !isRegionalGuidanceEnabled}
          />
        </GridItem>
        <GridItem position="relative" colSpan={2}>
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addInpaintMaskFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newInpaintMask')}
            isDisabled={isBusy || !isInpaintMaskEnabled}
          />
        </GridItem>
        <GridItem position="relative" colSpan={2}>
          <DndDropTarget
            dndTarget={newCanvasEntityFromImageDndTarget}
            dndTargetData={addResizedControlLayerFromImageDndTargetData}
            label={t('controlLayers.canvasContextMenu.newResizedControlLayer')}
            isDisabled={isBusy || !isControlLayerEnabled}
          />
        </GridItem>
      </Grid>
    </>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
