import { Box } from '@invoke-ai/ui-library';
import { useCanvasGenerationMode } from 'features/canvas/hooks/useCanvasGenerationMode';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const GenerationModeStatusText = () => {
  const generationMode = useCanvasGenerationMode();
  const { t } = useTranslation();

  const GENERATION_MODE_NAME_MAP = useMemo(
    () => ({
      txt2img: t('common.txt2img'),
      img2img: t('common.img2img'),
      inpaint: t('common.inpaint'),
      outpaint: t('common.outpaint'),
    }),
    [t]
  );

  return (
    <Box>
      {t('accessibility.mode')}: {generationMode ? GENERATION_MODE_NAME_MAP[generationMode] : '...'}
    </Box>
  );
};

export default memo(GenerationModeStatusText);
