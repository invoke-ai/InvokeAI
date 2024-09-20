import { Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvokeLogoIcon } from 'common/components/InvokeLogoIcon';
import { LOADING_SYMBOL, useHasImages } from 'features/gallery/hooks/useHasImages';
import { Trans, useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';

export const NoContentForViewer = () => {
  const hasImages = useHasImages();
  const { t } = useTranslation();

  if (hasImages === LOADING_SYMBOL) {
    return (
      // Blank bg w/ a spinner. The new user experience components below have an invoke logo, but it's not centered.
      // If we show the logo while loading, there is an awkward layout shift where the invoke logo moves a bit. Less
      // jarring to show a blank bg with a spinner - it will only be shown for a moment as we do the initial images
      // fetching.
      <Flex position="relative" width="full" height="full" alignItems="center" justifyContent="center">
        <Spinner label="Loading" color="grey" position="absolute" size="sm" width={8} height={8} right={4} bottom={4} />
      </Flex>
    );
  }

  if (hasImages) {
    return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  }

  return (
    <Flex flexDir="column" gap={4} alignItems="center" textAlign="center" maxW="600px">
      <InvokeLogoIcon w={40} h={40} />
      <Text fontSize="md" color="base.200" pt={16}>
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
