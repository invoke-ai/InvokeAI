import { Box } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { generationModeChanged } from 'features/canvas/store/canvasSlice';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getCanvasGenerationMode } from 'features/canvas/util/getCanvasGenerationMode';
import { useDebounce } from 'react-use';

const GENERATION_MODE_NAME_MAP = {
  txt2img: 'Text to Image',
  img2img: 'Image to Image',
  inpaint: 'Inpaint',
  outpaint: 'Inpaint',
};

export const useGenerationMode = () => {
  const dispatch = useAppDispatch();
  const canvasState = useAppSelector((state) => state.canvas);

  useDebounce(
    async () => {
      // Build canvas blobs
      const canvasBlobsAndImageData = await getCanvasData(canvasState);

      if (!canvasBlobsAndImageData) {
        return;
      }

      const { baseImageData, maskImageData } = canvasBlobsAndImageData;

      // Determine the generation mode
      const generationMode = getCanvasGenerationMode(
        baseImageData,
        maskImageData
      );

      dispatch(generationModeChanged(generationMode));
    },
    1000,
    [dispatch, canvasState, generationModeChanged]
  );
};

const GenerationModeStatusText = () => {
  const generationMode = useAppSelector((state) => state.canvas.generationMode);

  useGenerationMode();

  return (
    <Box>
      Mode: {generationMode ? GENERATION_MODE_NAME_MAP[generationMode] : '...'}
    </Box>
  );
};

export default GenerationModeStatusText;
