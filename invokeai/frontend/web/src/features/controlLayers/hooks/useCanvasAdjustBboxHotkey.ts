import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { bboxChangedFromCanvas } from 'features/controlLayers/store/canvasSlice';
import { selectMaskBlur } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasImageState,
  CanvasRectState,
  Rect,
} from 'features/controlLayers/store/types';
import { convertTransformedToOriginal } from 'features/controlLayers/util/coordinateTransform';
import {
  calculateMaskBoundsFromBitmap,
  transformMaskObjectsRelativeToBbox,
} from 'features/controlLayers/util/maskObjectTransform';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export const useCanvasAdjustBboxHotkey = () => {
  useAssertSingleton('useCanvasAdjustBboxHotkey');
  const dispatch = useAppDispatch();
  const canvasSlice = useAppSelector(selectCanvasSlice);
  const maskBlur = useAppSelector(selectMaskBlur);
  const isBusy = useCanvasIsBusy();
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

  const handleAdjustBbox = useCallback(() => {
    const maskBbox = calculateMaskBbox();
    if (!maskBbox) {
      return;
    }

    const padding = maskBlur + 8;
    const adjustedBbox: Rect = {
      x: maskBbox.x - padding,
      y: maskBbox.y - padding,
      width: maskBbox.width + padding * 2,
      height: maskBbox.height + padding * 2,
    };

    dispatch(bboxChangedFromCanvas(adjustedBbox));
  }, [dispatch, calculateMaskBbox, maskBlur]);

  const isAdjustBboxAllowed = useMemo(() => {
    const hasValidMasks = inpaintMasks.some((mask) => mask.isEnabled && mask.objects && mask.objects.length > 0);
    return hasValidMasks;
  }, [inpaintMasks]);

  useRegisteredHotkeys({
    id: 'adjustBbox',
    category: 'canvas',
    callback: handleAdjustBbox,
    options: { enabled: isAdjustBboxAllowed && !isBusy, preventDefault: true },
    dependencies: [isAdjustBboxAllowed, isBusy, handleAdjustBbox],
  });
};
