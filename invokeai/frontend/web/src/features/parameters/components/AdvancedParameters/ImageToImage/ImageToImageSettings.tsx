import { VStack } from '@chakra-ui/react';
import ImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';

import { useTranslation } from 'react-i18next';

export default function ImageToImageSettings() {
  const { t } = useTranslation();
  return (
    <VStack gap={2} alignItems="stretch">
      <ImageToImageStrength label={t('parameters.img2imgStrength')} />
      <ImageFit />
    </VStack>
  );
}
