import { Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { map } from 'es-toolkit/compat';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { StarterBundleButton } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterBundleButton';
import { StarterBundleTooltipContentCompact } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterBundleTooltipContentCompact';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiLinkBold, PiStarBold } from 'react-icons/pi';
import { SiHuggingface } from 'react-icons/si';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';

import { LaunchpadButton } from './LaunchpadButton';

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
              <LaunchpadButton
                onClick={navigateToUrlTab}
                icon={PiLinkBold}
                title={t('modelManager.urlOrLocalPath')}
                description={t('modelManager.launchpad.urlDescription')}
              />
              <LaunchpadButton
                onClick={navigateToHuggingFaceTab}
                icon={SiHuggingface}
                title={t('modelManager.huggingFace')}
                description={t('modelManager.launchpad.huggingFaceDescription')}
              />
              <LaunchpadButton
                onClick={navigateToScanFolderTab}
                icon={PiFolderOpenBold}
                title={t('modelManager.scanFolder')}
                description={t('modelManager.launchpad.scanFolderDescription')}
              />
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
