import { VStack } from '@chakra-ui/react';

import ImageToImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';

import { useTranslation } from 'react-i18next';
import InitialImagePreview from './InitialImagePreview';
import InitialImageButtons from 'common/components/ImageToImageButtons';

export default function ImageToImageSettings() {
  const { t } = useTranslation();
  return (
    <VStack gap={2} w="full" alignItems="stretch">
      <InitialImageButtons />
      <InitialImagePreview />
      <ImageToImageStrength />
      <ImageToImageFit />
    </VStack>
  );
}
