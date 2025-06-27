import { Box, Button, Flex, Grid, Heading, Icon, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStarterBundleInstall } from 'features/modelManagerV2/hooks/useStarterBundleInstall';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiLinkBold, PiStarBold } from 'react-icons/pi';
import { SiHuggingface } from 'react-icons/si';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';

export const LaunchpadForm = memo(() => {
  const { t } = useTranslation();
  const { installBundle } = useStarterBundleInstall();
  const { data: starterModelsData } = useGetStarterModelsQuery();

  // Function to install models from a bundle
  const handleBundleInstall = useCallback(
    (bundleName: string) => {
      if (!starterModelsData?.starter_bundles) {
        return;
      }

      const bundle = starterModelsData.starter_bundles[bundleName];
      if (!bundle) {
        return;
      }

      installBundle(bundle, bundleName);
    },
    [starterModelsData, installBundle]
  );

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

  const handleSD15BundleClick = useCallback(() => {
    handleBundleInstall('sd-1');
  }, [handleBundleInstall]);

  const handleSDXLBundleClick = useCallback(() => {
    handleBundleInstall('sdxl');
  }, [handleBundleInstall]);

  const handleFluxBundleClick = useCallback(() => {
    handleBundleInstall('flux');
  }, [handleBundleInstall]);

  return (
    <Flex flexDir="column" height="100%" gap={3}>
      <ScrollableContent>
        <Flex flexDir="column" gap={6} p={3}>
          {/* Welcome Section */}
          <Box>
            <Heading size="md" mb={1}>
              {t('modelManager.launchpad.welcome')}
            </Heading>
            <Text color="base.300" fontSize="sm">
              {t('modelManager.launchpad.description')}
            </Text>
          </Box>
          {/* Manual Installation Options */}
          <Box>
            <Heading size="sm" mb={2}>
              {t('modelManager.launchpad.manualInstall')}
            </Heading>
            <Grid templateColumns="repeat(auto-fit, minmax(280px, 1fr))" gap={3}>
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
                <Text fontSize="sm" lineHeight="1.4" flex="1" whiteSpace="normal" wordBreak="break-word">
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
          </Box>
          {/* Recommended Section */}
          <Box>
            <Heading size="sm" mb={2}>
              {t('modelManager.launchpad.recommendedModels')}
            </Heading>
            <Flex flexDir="column" gap={2}>
              {/* Starter Model Bundles - More Prominent */}
              <Box>
                <Heading size="xs" color="base.100" mb={1}>
                  {t('modelManager.launchpad.quickStart')}
                </Heading>
                <Text fontSize="xs" color="base.300" mb={2}>
                  {t('modelManager.launchpad.bundleDescription')}
                </Text>
                <Grid templateColumns="repeat(auto-fit, minmax(180px, 1fr))" gap={2}>
                  <Button
                    onClick={handleSD15BundleClick}
                    variant="outline"
                    h="auto"
                    minH={10}
                    p={3}
                    textAlign="center"
                    justifyContent="center"
                    alignItems="center"
                    borderRadius="lg"
                    whiteSpace="normal"
                  >
                    <Text fontSize="sm" fontWeight="bold" noOfLines={1}>
                      {t('modelManager.launchpad.stableDiffusion15')}
                    </Text>
                  </Button>
                  <Button
                    onClick={handleSDXLBundleClick}
                    variant="outline"
                    h="auto"
                    minH={10}
                    p={3}
                    textAlign="center"
                    justifyContent="center"
                    alignItems="center"
                    borderRadius="lg"
                    whiteSpace="normal"
                  >
                    <Text fontSize="sm" fontWeight="bold" noOfLines={1}>
                      {t('modelManager.launchpad.sdxl')}
                    </Text>
                  </Button>
                  <Button
                    onClick={handleFluxBundleClick}
                    variant="outline"
                    h="auto"
                    minH={10}
                    p={3}
                    textAlign="center"
                    justifyContent="center"
                    alignItems="center"
                    borderRadius="lg"
                    whiteSpace="normal"
                  >
                    <Text fontSize="sm" fontWeight="bold" noOfLines={1}>
                      {t('modelManager.launchpad.fluxDev')}
                    </Text>
                  </Button>
                </Grid>
              </Box>
              {/* Browse All - Simple Link */}
              <Box pt={1} borderTop="1px solid" borderColor="base.700">
                <Text fontSize="xs" color="base.400" mb={1}>
                  {t('modelManager.launchpad.browseAll')}
                </Text>
                <Button
                  onClick={navigateToStarterModelsTab}
                  variant="link"
                  color="invokeBlue.300"
                  fontSize="sm"
                  fontWeight="medium"
                  p={0}
                  h="auto"
                  leftIcon={<PiStarBold size={16} />}
                >
                  {t('modelManager.launchpad.exploreStarter')}
                </Button>
              </Box>
            </Flex>
          </Box>
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

LaunchpadForm.displayName = 'LaunchpadForm';
