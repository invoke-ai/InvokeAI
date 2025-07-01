import { Button, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { bboxChangedFromCanvas } from 'features/controlLayers/store/canvasSlice';
import { selectMaskBlur } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { Rect } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const InpaintMaskBboxAdjuster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasSlice = useAppSelector(selectCanvasSlice);
  const maskBlur = useAppSelector(selectMaskBlur);

  // Get all inpaint mask entities
  const inpaintMasks = canvasSlice.inpaintMasks.entities;

  // Calculate the bounding box that contains all inpaint masks
  const calculateMaskBbox = useCallback((): Rect | null => {
    if (inpaintMasks.length === 0) {
      return null;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    // Iterate through all inpaint masks to find the overall bounds
    for (const mask of inpaintMasks) {
      if (!mask.isEnabled || mask.objects.length === 0) {
        continue;
      }

      // Calculate bounds for this mask's objects
      for (const obj of mask.objects) {
        let objMinX = 0;
        let objMinY = 0;
        let objMaxX = 0;
        let objMaxY = 0;

        if (obj.type === 'rect') {
          objMinX = mask.position.x + obj.rect.x;
          objMinY = mask.position.y + obj.rect.y;
          objMaxX = objMinX + obj.rect.width;
          objMaxY = objMinY + obj.rect.height;
        } else if (
          obj.type === 'brush_line' ||
          obj.type === 'brush_line_with_pressure' ||
          obj.type === 'eraser_line' ||
          obj.type === 'eraser_line_with_pressure'
        ) {
          // For lines, find the min/max points
          for (let i = 0; i < obj.points.length; i += 2) {
            const x = mask.position.x + (obj.points[i] ?? 0);
            const y = mask.position.y + (obj.points[i + 1] ?? 0);

            if (i === 0) {
              objMinX = objMaxX = x;
              objMinY = objMaxY = y;
            } else {
              objMinX = Math.min(objMinX, x);
              objMinY = Math.min(objMinY, y);
              objMaxX = Math.max(objMaxX, x);
              objMaxY = Math.max(objMaxY, y);
            }
          }
          // Add stroke width to account for line thickness
          const strokeRadius = (obj.strokeWidth ?? 50) / 2;
          objMinX -= strokeRadius;
          objMinY -= strokeRadius;
          objMaxX += strokeRadius;
          objMaxY += strokeRadius;
        } else if (obj.type === 'image') {
          // Image objects are positioned at the entity's position
          objMinX = mask.position.x;
          objMinY = mask.position.y;
          objMaxX = objMinX + obj.image.width;
          objMaxY = objMinY + obj.image.height;
        }

        // Update overall bounds
        minX = Math.min(minX, objMinX);
        minY = Math.min(minY, objMinY);
        maxX = Math.max(maxX, objMaxX);
        maxY = Math.max(maxY, objMaxY);
      }
    }

    // If no valid bounds found, return null
    if (minX === Infinity || minY === Infinity || maxX === -Infinity || maxY === -Infinity) {
      return null;
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }, [inpaintMasks]);

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
