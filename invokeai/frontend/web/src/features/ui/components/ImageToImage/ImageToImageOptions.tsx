import { Flex } from '@chakra-ui/react';
import ImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import ParametersAccordion from 'features/parameters/components/ParametersAccordion';

import { useTranslation } from 'react-i18next';

export default function ImageToImageOptions() {
  const { t } = useTranslation();
  const imageToImageAccordionItems = {
    imageToImage: {
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: (
        <Flex gap={2} flexDir="column">
          <ImageToImageStrength
            label={t('parameters.img2imgStrength')}
            styleClass="main-settings-block image-to-image-strength-main-option"
          />
          <ImageFit />
        </Flex>
      ),
    },
  };
  return <ParametersAccordion accordionInfo={imageToImageAccordionItems} />;
}
