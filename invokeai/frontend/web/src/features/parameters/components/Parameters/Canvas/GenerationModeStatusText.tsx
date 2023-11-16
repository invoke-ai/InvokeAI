import { Box } from '@chakra-ui/react';
import { useCanvasGenerationMode } from 'features/canvas/hooks/useCanvasGenerationMode';
import { memo } from 'react';
import { t } from 'i18next';

const GENERATION_MODE_NAME_MAP = {
  txt2img: t('common.txt2img'),
  img2img: t('common.img2img'),
  inpaint: t('common.inpaint'),
  outpaint: t('common.outpaint'),
};

const GenerationModeStatusText = () => {
  const generationMode = useCanvasGenerationMode();

  return (
    <Box>
      {t('accessibility.mode')}:{' '}
      {generationMode ? GENERATION_MODE_NAME_MAP[generationMode] : '...'}
    </Box>
  );
};

export default memo(GenerationModeStatusText);
