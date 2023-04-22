import { Flex, Image, VStack } from '@chakra-ui/react';
import ImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';

import { useTranslation } from 'react-i18next';
import InitialImagePreview from './InitialImagePreview';

export default function ImageToImageSettings() {
  const { t } = useTranslation();
  return (
    <VStack gap={2} alignItems="stretch">
      <InitialImagePreview />
      <ImageToImageStrength label={t('parameters.img2imgStrength')} />
      <ImageFit />
    </VStack>
  );
}
