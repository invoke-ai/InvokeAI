import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { UPSCALE_SCALE } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const UpscaleSizeDetails = () => {
  const { t } = useTranslation();
  const { upscaleInitialImage } = useAppSelector((s) => s.upscale);

  const outputSizeText = useMemo(() => {
    if (upscaleInitialImage) {
      return `${t('upscaling.outputImageSize')}: ${upscaleInitialImage.width * UPSCALE_SCALE} ${t('upscaling.x')} ${upscaleInitialImage.height * UPSCALE_SCALE}`;
    }
  }, [upscaleInitialImage, t]);

  if (!outputSizeText || !upscaleInitialImage) {
    return <></>;
  }

  return (
    <Flex direction="column">
      <Text variant="subtext" fontWeight="bold">
        {t('upscaling.currentImageSize')}: {upscaleInitialImage.width} {t('upscaling.x')} {upscaleInitialImage.height}
      </Text>
      <Text variant="subtext" fontWeight="bold">
        {outputSizeText}
      </Text>
    </Flex>
  );
};
