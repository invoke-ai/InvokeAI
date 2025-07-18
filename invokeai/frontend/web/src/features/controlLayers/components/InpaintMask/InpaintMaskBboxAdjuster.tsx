import { Button, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
import { maskObjectsToBitmap } from 'features/controlLayers/util/bitmapToMaskObjects';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const InpaintMaskBboxAdjuster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasSlice = useAppSelector(selectCanvasSlice);
  const maskBlur = useAppSelector(selectMaskBlur);

  const inpaintMasks = canvasSlice.inpaintMasks.entities;
  const bboxRect = canvasSlice.bbox.rect;

  // Calculate the bounding box that contains all inpaint masks
  const calculateMaskBbox = useCallback((): Rect | null => {
    if (inpaintMasks.length === 0) {
      return null;
    }

    const canvasWidth = bboxRect.width;
    const canvasHeight = bboxRect.height;

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

      for (const obj of mask.objects) {
        if (obj.type === 'rect') {
          const adjustedObj = {
            ...obj,
            rect: {
              ...obj.rect,
              x: obj.rect.x + mask.position.x - bboxRect.x,
              y: obj.rect.y + mask.position.y - bboxRect.y,
            },
          };
          allObjects.push(adjustedObj);
        } else if (
          obj.type === 'brush_line' ||
          obj.type === 'brush_line_with_pressure' ||
          obj.type === 'eraser_line' ||
          obj.type === 'eraser_line_with_pressure'
        ) {
          const adjustedPoints: number[] = [];
          for (let i = 0; i < obj.points.length; i += 2) {
            adjustedPoints.push((obj.points[i] ?? 0) + mask.position.x - bboxRect.x);
            adjustedPoints.push((obj.points[i + 1] ?? 0) + mask.position.y - bboxRect.y);
          }
          const adjustedObj = {
            ...obj,
            points: adjustedPoints,
          };
          allObjects.push(adjustedObj);
        } else if (obj.type === 'image') {
          // For image objects, we need to handle them differently since they don't have rect or points
          continue;
        }
      }
    }

    if (allObjects.length === 0) {
      return null;
    }

    // Render the consolidated mask to a bitmap
    const bitmap = maskObjectsToBitmap(allObjects, canvasWidth, canvasHeight);
    const { width, height, data } = bitmap;

    let maskMinX = width;
    let maskMinY = height;
    let maskMaxX = 0;
    let maskMaxY = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = (y * width + x) * 4;
        const alpha = data[pixelIndex + 3] ?? 0;

        if (alpha > 0) {
          maskMinX = Math.min(maskMinX, x);
          maskMinY = Math.min(maskMinY, y);
          maskMaxX = Math.max(maskMaxX, x);
          maskMaxY = Math.max(maskMaxY, y);
        }
      }
    }

    if (maskMinX >= maskMaxX || maskMinY >= maskMaxY) {
      return null;
    }

    maskMinX = Math.max(0, maskMinX);
    maskMinY = Math.max(0, maskMinY);
    maskMaxX = Math.min(width - 1, maskMaxX);
    maskMaxY = Math.min(height - 1, maskMaxY);

    return {
      x: bboxRect.x + maskMinX,
      y: bboxRect.y + maskMinY,
      width: maskMaxX - maskMinX + 1,
      height: maskMaxY - maskMinY + 1,
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

  const hasValidMasks = inpaintMasks.some((mask) => mask.isEnabled && mask.objects.length > 0);
  if (!hasValidMasks) {
    return null;
  }

  return (
    <Flex w="full" ps={2} pe={2} pb={1} gap={2}>
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<Icon as={PiCropBold} boxSize={3} />}
        onClick={handleAdjustBbox}
        flex={1}
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
