import { Box, Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { flattenStarterModel, useBuildModelInstallArg } from 'features/modelManagerV2/hooks/useBuildModelsToInstall';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import { toast } from 'features/toast/toast';
import { flatMap, negate, uniqWith } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiLinkBold, PiStarBold } from 'react-icons/pi';
import { SiHuggingface } from "react-icons/si";
import { useGetStarterModelsQuery, useInstallModelMutation } from 'services/api/endpoints/models';

export const LaunchpadForm = memo(() => {
  const { t } = useTranslation();
  const [installModel] = useInstallModelMutation();
  const { getIsInstalled, buildModelInstallArg } = useBuildModelInstallArg();
  const { data: starterModelsData } = useGetStarterModelsQuery();
  // Function to install models from a bundle
  const installBundle = useCallback((bundleName: string) => {
    if (!starterModelsData?.starter_bundles) {
      return;
    }
    
    const bundle = starterModelsData.starter_bundles[bundleName];
    if (!bundle) {
      return;
    }

    // Flatten the models and remove duplicates, which is expected as models can have the same dependencies
    const flattenedModels = flatMap(bundle, flattenStarterModel);
    const uniqueModels = uniqWith(
      flattenedModels,
      (m1, m2) => m1.source === m2.source || (m1.name === m2.name && m1.base === m2.base && m1.type === m2.type)
    );
    // We want to install models that are not installed and skip models that are already installed
    const install = uniqueModels.filter(negate(getIsInstalled)).map(buildModelInstallArg);
    const skip = uniqueModels.filter(getIsInstalled).map(buildModelInstallArg);

    if (install.length === 0) {
      toast({
        status: 'info',
        title: t('modelManager.bundleAlreadyInstalled', { bundleName }),
        description: t('modelManager.allModelsAlreadyInstalled'),
      });
      return;
    }

    // Install all models in the bundle
    install.forEach(installModel);
    
    let description = t('modelManager.installingXModels', { count: install.length });
    if (skip.length > 1) {
      description += t('modelManager.skippingXDuplicates', { count: skip.length - 1 });
    }
    
    toast({
      status: 'info',
      title: t('modelManager.installingBundle'),
      description,
    });
  }, [starterModelsData, getIsInstalled, buildModelInstallArg, installModel, t]);

  const navigateToUrlTab = useCallback(() => {
    $installModelsTab.set(1); // URL/Local Path tab (now index 1)
  }, []);

  const navigateToHuggingFaceTab = useCallback(() => {
    $installModelsTab.set(2); // HuggingFace tab (now index 2)
  }, []);

  const navigateToScanFolderTab = useCallback(() => {
    $installModelsTab.set(3); // Scan Folder tab (now index 3)
  }, []);
  
  const navigateToStarterModelsTab = useCallback(() => {
    $installModelsTab.set(4); // Starter Models tab (now index 4)
  }, []);
  const handleSD15BundleClick = useCallback(() => {
    installBundle('sd-1');
  }, [installBundle]);

  const handleSDXLBundleClick = useCallback(() => {
    installBundle('sdxl');
  }, [installBundle]);

  const handleFluxBundleClick = useCallback(() => {
    installBundle('flux');
  }, [installBundle]);  return (
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
        </Heading>        <Grid templateColumns="repeat(auto-fit, minmax(280px, 1fr))" gap={3}>
          <LaunchpadCard
            title={t('modelManager.urlOrLocalPath')}
            description={t('modelManager.launchpad.urlDescription')}
            icon={<PiLinkBold size={24} />}
            onClick={navigateToUrlTab}
          />
          <LaunchpadCard
            title={t('modelManager.huggingFace')}
            description={t('modelManager.launchpad.huggingFaceDescription')}
            icon={<SiHuggingface  size={24} />}
            onClick={navigateToHuggingFaceTab}
          />
          <LaunchpadCard
            title={t('modelManager.scanFolder')}
            description={t('modelManager.launchpad.scanFolderDescription')}
            icon={<PiFolderOpenBold size={24} />}
            onClick={navigateToScanFolderTab}
          />
        </Grid>
      </Box>      {/* Recommended Section */}
      <Box>
        <Heading size="sm" mb={2}>
          {t('modelManager.launchpad.recommendedModels')}
        </Heading>
        <Flex flexDir="column" gap={2}>          {/* Starter Model Bundles - More Prominent */}
          <Box>
            <Heading size="xs" color="base.100" mb={1}>
              {t('modelManager.launchpad.quickStart')}
            </Heading>
            <Text fontSize="xs" color="base.300" mb={2}>
              {t('modelManager.launchpad.bundleDescription')}
            </Text>            <Grid templateColumns="repeat(auto-fit, minmax(180px, 1fr))" gap={2}>
              <LaunchpadBundleCard
                title="Stable Diffusion 1.5"
                onClick={handleSD15BundleClick}
              />
              <LaunchpadBundleCard
                title="SDXL"
                onClick={handleSDXLBundleClick}
              />
              <LaunchpadBundleCard
                title="FLUX.1 [dev]"
                onClick={handleFluxBundleClick}
              />
            </Grid>
          </Box>            {/* Browse All - Simple Link */}
          <Box pt={1} borderTop="1px solid" borderColor="base.700">
            <Text fontSize="xs" color="base.400" mb={1}>
              {t('modelManager.launchpad.browseAll')}
            </Text>
            <Button
              onClick={navigateToStarterModelsTab}
              variant="link"              color="invokeBlue.300"
              fontSize="sm"
              fontWeight="medium"
              p={0}
              h="auto"
              leftIcon={<PiStarBold size={16} />}
              _hover={{
                color: "invokeBlue.200",
                textDecoration: "underline"
              }}
            >
              {t('modelManager.launchpad.exploreStarter')}
            </Button>          </Box>        </Flex>      </Box>
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

LaunchpadForm.displayName = 'LaunchpadForm';

interface LaunchpadCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  onClick: () => void;
  variant?: 'default' | 'featured';
}

const LaunchpadCard = memo(({ title, description, icon, onClick, variant = 'default' }: LaunchpadCardProps) => {
  return (    <Button
      onClick={onClick}
      variant="outline"
      h="auto"
      minH="50px"
      p={4}
      borderWidth={variant === 'featured' ? 2 : 1}
      borderColor={variant === 'featured' ? 'invokeBlue.300' : 'base.700'}
      bg={variant === 'featured' ? 'invokeBlue.900' : 'base.850'}
      _hover={{
        bg: variant === 'featured' ? 'invokeBlue.800' : 'base.800',
        borderColor: variant === 'featured' ? 'invokeBlue.200' : 'base.600',
        transform: 'translateY(-2px)',
      }}
      _active={{
        transform: 'translateY(0px)',
      }}
      transition="all 0.2s"
      cursor="pointer"
      textAlign="left"
      justifyContent="flex-start"
      alignItems="flex-start"
      flexDir="column"
      gap={2}
      borderRadius="lg"
      whiteSpace="normal"
    >
      <Flex alignItems="center" gap={2} w="full">
        <Box color={variant === 'featured' ? 'invokeBlue.200' : 'base.300'} flexShrink={0}>
          {icon}
        </Box>
        <Heading size="sm" color={variant === 'featured' ? 'invokeBlue.50' : 'base.100'} noOfLines={2}>
          {title}
        </Heading>
      </Flex>
      <Text 
        fontSize="sm" 
        color={variant === 'featured' ? 'invokeBlue.200' : 'base.400'} 
        lineHeight="1.4"
        flex="1"
        whiteSpace="normal"
        wordBreak="break-word"
      >
        {description}
      </Text>
    </Button>
  );
});

LaunchpadCard.displayName = 'LaunchpadCard';

interface LaunchpadBundleCardProps {
  title: string;
  onClick: () => void;
}

const LaunchpadBundleCard = memo(({ title, onClick }: LaunchpadBundleCardProps) => {
  return (
    <Button
      onClick={onClick}
      variant="outline"
      h="auto"
      minH="40px"
      p={3}
      borderWidth={2}
      borderColor="invokeBlue.400"
      bg="invokeBlue.950"
      _hover={{
        bg: "invokeBlue.900",
        borderColor: "invokeBlue.300",
        transform: "translateY(-2px)",
        boxShadow: "0 4px 20px rgba(66, 153, 225, 0.15)",
      }}
      _active={{
        transform: "translateY(0px)",
      }}
      transition="all 0.2s"
      cursor="pointer"
      textAlign="center"
      justifyContent="center"
      alignItems="center"      borderRadius="lg"
      whiteSpace="normal"
    >
      <Text fontSize="sm" fontWeight="bold" color="invokeBlue.100" noOfLines={1}>
        {title}
      </Text>
    </Button>
  );
});

LaunchpadBundleCard.displayName = 'LaunchpadBundleCard';
