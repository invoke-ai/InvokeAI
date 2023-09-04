import { Box } from '@chakra-ui/react';
import { useCanvasGenerationMode } from 'features/canvas/hooks/useCanvasGenerationMode';
import { memo } from 'react';

const GENERATION_MODE_NAME_MAP = {
  txt2img: 'Text to Image',
  img2img: 'Image to Image',
  inpaint: 'Inpaint',
  outpaint: 'Inpaint',
};

const GenerationModeStatusText = () => {
  const generationMode = useCanvasGenerationMode();

  return (
    <Box>
      Mode: {generationMode ? GENERATION_MODE_NAME_MAP[generationMode] : '...'}
    </Box>
  );
};

export default memo(GenerationModeStatusText);
