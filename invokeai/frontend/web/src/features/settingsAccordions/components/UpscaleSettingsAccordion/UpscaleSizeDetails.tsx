import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const UpscaleSizeDetails = () => {
  const { t } = useTranslation();
  const { upscaleInitialImage, scale } = useAppSelector((s) => s.upscale);

  const outputSizeText = useMemo(() => {
    if (upscaleInitialImage && scale) {
      return `${t('upscaling.outputImageSize')}: ${upscaleInitialImage.width * scale} ${t('upscaling.x')} ${upscaleInitialImage.height * scale}`;
    }
  }, [upscaleInitialImage, scale, t]);

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
