import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { LOADING_SYMBOL, useHasImages } from 'features/gallery/hooks/useHasImages';
import InvokeSymbol from 'public/assets/images/invoke-symbol-char-lrg.svg';
import { Trans, useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import Loading from '../../../../common/components/Loading/Loading';

export const NoContentForViewer = () => {
  const hasImages = useHasImages();
  const { t } = useTranslation();

  if (hasImages === LOADING_SYMBOL) {
    return <Loading />;
  }

  if (hasImages) {
    return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  }

  return (
    <Flex flexDir="column" gap={4} alignItems="center" textAlign="center" maxW="600px">
      <Image src={InvokeSymbol} w="8rem" h="8rem" />
      <Text fontSize="md" color="base.200">
        <Trans
          i18nKey="newUserExperience.toGetStarted"
          components={{
            StrongComponent: <Text as="span" color="white" fontSize="md" fontWeight="semibold" />,
          }}
        />
      </Text>

      <Text fontSize="md" color="base.200">
        <Trans
          i18nKey="newUserExperience.gettingStartedSeries"
          components={{
            LinkComponent: (
              <Text
                as="a"
                color="white"
                fontSize="md"
                fontWeight="semibold"
                href="https://www.youtube.com/@invokeai/videos"
                target="_blank"
              />
            ),
          }}
        />
      </Text>
    </Flex>
  );
};
