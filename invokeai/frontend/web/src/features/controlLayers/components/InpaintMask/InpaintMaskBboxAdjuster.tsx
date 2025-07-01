import { Button, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxChangedFromCanvas } from 'features/controlLayers/store/canvasSlice';
import { selectMaskBlur } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { 
  Rect,
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasRectState,
  CanvasImageState,
} from 'features/controlLayers/store/types';
import { transformMaskObjectsRelativeToBbox, calculateMaskBoundsFromBitmap } from 'features/controlLayers/util/maskObjectTransform';
import { convertTransformedToOriginal } from 'features/controlLayers/util/coordinateTransform';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const InpaintMaskBboxAdjuster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasSlice = useAppSelector(selectCanvasSlice);
  const maskBlur = useAppSelector(selectMaskBlur);

  // Get all inpaint mask entities and bbox
  const inpaintMasks = canvasSlice.inpaintMasks.entities;
  const bboxRect = canvasSlice.bbox.rect;

  // Calculate the bounding box that contains all inpaint masks
  const calculateMaskBbox = useCallback((): Rect | null => {
    if (inpaintMasks.length === 0) {
      return null;
    }

    // Collect all mask objects from enabled masks
    const allObjects: (
      | CanvasBrushLineState
      | CanvasBrushLineWithPressureState
      | CanvasEraserLineState
      | CanvasEraserLineWithPressureState
      | CanvasRectState
      | CanvasImageState
    )[] = [];
    
    for (const mask of inpaintMasks) {
      if (!mask.isEnabled || !mask.objects || mask.objects.length === 0) {
        continue;
      }

      // Transform objects to be relative to the bbox
      const transformedObjects = transformMaskObjectsRelativeToBbox(mask.objects, bboxRect);
      // Convert back to original types for compatibility
      const originalObjects = transformedObjects.map(convertTransformedToOriginal);
      allObjects.push(...originalObjects);
    }

    if (allObjects.length === 0) {
      return null;
    }

    // Calculate bounds from the rendered bitmap for accurate results
    const maskBounds = calculateMaskBoundsFromBitmap(allObjects, bboxRect.width, bboxRect.height);
    
    if (!maskBounds) {
      return null;
    }

    // Convert back to world coordinates relative to the bbox
    return {
      x: bboxRect.x + maskBounds.x,
      y: bboxRect.y + maskBounds.y,
      width: maskBounds.width,
      height: maskBounds.height,
    };
  }, [inpaintMasks, bboxRect]);

  const maskBbox = useMemo(() => calculateMaskBbox(), [calculateMaskBbox]);

  const handleAdjustBbox = useCallback(() => {
    if (!maskBbox) {
      return;
    }

    // Add padding based on maskblur setting + 8px
    const padding = maskBlur + 8;
    const adjustedBbox: Rect = {
      x: maskBbox.x - padding,
      y: maskBbox.y - padding,
      width: maskBbox.width + padding * 2,
      height: maskBbox.height + padding * 2,
    };

    dispatch(bboxChangedFromCanvas(adjustedBbox));
  }, [dispatch, maskBbox, maskBlur]);

  // Only show if there are enabled inpaint masks with objects
  const hasValidMasks = inpaintMasks.some((mask) => mask.isEnabled && mask.objects.length > 0);
  if (!hasValidMasks) {
    return null;
  }

  return (
    <Flex w="full" ps={2} pe={2} pb={1}>
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<Icon as={PiCropBold} boxSize={3} />}
        onClick={handleAdjustBbox}
        w="full"
        justifyContent="flex-start"
        h={6}
        fontSize="xs"
      >
        <Text fontSize="xs" fontWeight="medium">
          {t('controlLayers.adjustBboxToMasks')}
        </Text>
      </Button>
    </Flex>
  );
});

InpaintMaskBboxAdjuster.displayName = 'InpaintMaskBboxAdjuster';
