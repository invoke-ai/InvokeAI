import { Button, Flex, Grid, Heading, Icon, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { map } from 'es-toolkit/compat';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { StarterBundleButton } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterBundle';
import { StarterBundleTooltipContentCompact } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterBundleTooltipContentCompact';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiLinkBold, PiStarBold } from 'react-icons/pi';
import { SiHuggingface } from 'react-icons/si';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';

export const LaunchpadForm = memo(() => {
  const { t } = useTranslation();
  const { data: starterModelsData } = useGetStarterModelsQuery();

  const navigateToUrlTab = useCallback(() => {
    setInstallModelsTabByName('urlOrLocal');
  }, []);

  const navigateToHuggingFaceTab = useCallback(() => {
    setInstallModelsTabByName('huggingface');
  }, []);

  const navigateToScanFolderTab = useCallback(() => {
    setInstallModelsTabByName('scanFolder');
  }, []);

  const navigateToStarterModelsTab = useCallback(() => {
    setInstallModelsTabByName('starterModels');
  }, []);

  return (
    <Flex flexDir="column" height="100%" gap={3}>
      <ScrollableContent>
        <Flex flexDir="column" gap={6} p={3}>
          {/* Welcome Section */}
          <Flex flexDir="column" gap={2} alignItems="flex-start">
            <Heading size="md">{t('modelManager.launchpad.welcome')}</Heading>
            <Text color="base.300">{t('modelManager.launchpad.description')}</Text>
          </Flex>
          {/* Manual Installation Options */}
          <Flex flexDir="column" gap={2} alignItems="flex-start">
            <Heading size="sm">{t('modelManager.launchpad.manualInstall')}</Heading>
            <Grid templateColumns="repeat(auto-fit, minmax(280px, 1fr))" gap={3} w="full">
              <Button
                onClick={navigateToUrlTab}
                variant="outline"
                p={4}
                textAlign="left"
                flexDir="column"
                gap={2}
                h="unset"
              >
                <Flex alignItems="center" gap={4} w="full">
                  <Icon as={PiLinkBold} boxSize={8} color="base.300" />
                  <Heading size="sm" color="base.100">
                    {t('modelManager.urlOrLocalPath')}
                  </Heading>
                </Flex>
                <Text lineHeight="1.4" flex="1" whiteSpace="normal" wordBreak="break-word">
                  {t('modelManager.launchpad.urlDescription')}
                </Text>
              </Button>
              <Button
                onClick={navigateToHuggingFaceTab}
                variant="outline"
                p={4}
                textAlign="left"
                flexDir="column"
                gap={2}
                h="unset"
              >
                <Flex alignItems="center" gap={4} w="full">
                  <Icon as={SiHuggingface} boxSize={8} color="base.300" />
                  <Heading size="sm" color="base.100">
                    {t('modelManager.huggingFace')}
                  </Heading>
                </Flex>
                <Text
                  fontSize="sm"
                  color="base.400"
                  lineHeight="1.4"
                  flex="1"
                  whiteSpace="normal"
                  wordBreak="break-word"
                >
                  {t('modelManager.launchpad.huggingFaceDescription')}
                </Text>
              </Button>
              <Button
                onClick={navigateToScanFolderTab}
                variant="outline"
                p={4}
                textAlign="left"
                flexDir="column"
                gap={2}
                h="unset"
              >
                <Flex alignItems="center" gap={4} w="full">
                  <Icon as={PiFolderOpenBold} boxSize={8} color="base.300" />
                  <Heading size="sm" color="base.100">
                    {t('modelManager.scanFolder')}
                  </Heading>
                </Flex>
                <Text
                  fontSize="sm"
                  color="base.400"
                  lineHeight="1.4"
                  flex="1"
                  whiteSpace="normal"
                  wordBreak="break-word"
                >
                  {t('modelManager.launchpad.scanFolderDescription')}
                </Text>
              </Button>
            </Grid>
          </Flex>
          {/* Recommended Section */}
          {starterModelsData && (
            <Flex flexDir="column" gap={2} alignItems="flex-start">
              <Heading size="sm">{t('modelManager.launchpad.recommendedModels')}</Heading>
              {/* Starter Model Bundles - More Prominent */}
              <Text color="base.300">{t('modelManager.launchpad.bundleDescription')}</Text>
              <Grid templateColumns="repeat(auto-fit, minmax(180px, 1fr))" gap={2} w="full">
                {map(starterModelsData.starter_bundles, (bundle) => (
                  <StarterBundleButton
                    size="md"
                    tooltip={<StarterBundleTooltipContentCompact bundle={bundle} />}
                    key={bundle.name}
                    bundle={bundle}
                    variant="outline"
                    p={4}
                    h="unset"
                  />
                ))}
              </Grid>
              {/* Browse All - Simple Link */}
              <Button
                onClick={navigateToStarterModelsTab}
                variant="link"
                size="sm"
                leftIcon={<PiStarBold />}
                colorScheme="invokeBlue"
              >
                {t('modelManager.launchpad.exploreStarter')}
              </Button>
            </Flex>
          )}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

LaunchpadForm.displayName = 'LaunchpadForm';
