import { useAppSelector } from 'app/store/storeHooks';
import type { GenerationMode } from 'features/canvas/store/canvasTypes';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getCanvasGenerationMode } from 'features/canvas/util/getCanvasGenerationMode';
import { useEffect, useState } from 'react';
import { useDebounce } from 'react-use';

export const useCanvasGenerationMode = () => {
  const layerState = useAppSelector((s) => s.canvas.layerState);

  const boundingBoxCoordinates = useAppSelector((s) => s.canvas.boundingBoxCoordinates);
  const boundingBoxDimensions = useAppSelector((s) => s.canvas.boundingBoxDimensions);
  const isMaskEnabled = useAppSelector((s) => s.canvas.isMaskEnabled);

  const shouldPreserveMaskedArea = useAppSelector((s) => s.canvas.shouldPreserveMaskedArea);
  const [generationMode, setGenerationMode] = useState<GenerationMode | undefined>();

  useEffect(() => {
    setGenerationMode(undefined);
  }, [layerState, boundingBoxCoordinates, boundingBoxDimensions, isMaskEnabled, shouldPreserveMaskedArea]);

  useDebounce(
    async () => {
      // Build canvas blobs
      const canvasBlobsAndImageData = await getCanvasData(
        layerState,
        boundingBoxCoordinates,
        boundingBoxDimensions,
        isMaskEnabled,
        shouldPreserveMaskedArea
      );

      if (!canvasBlobsAndImageData) {
        return;
      }

      const { baseImageData, maskImageData } = canvasBlobsAndImageData;

      // Determine the generation mode
      const generationMode = getCanvasGenerationMode(baseImageData, maskImageData);

      setGenerationMode(generationMode);
    },
    1000,
    [layerState, boundingBoxCoordinates, boundingBoxDimensions, isMaskEnabled, shouldPreserveMaskedArea]
  );

  return generationMode;
};
