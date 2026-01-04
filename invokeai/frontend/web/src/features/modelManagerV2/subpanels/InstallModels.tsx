import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $installModelsTabIndex } from 'features/modelManagerV2/store/installModelsStore';
import { StarterModelsForm } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterModelsForm';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCubeBold, PiFolderOpenBold, PiLinkSimpleBold, PiShootingStarBold } from 'react-icons/pi';
import { SiHuggingface } from 'react-icons/si';

import { HuggingFaceForm } from './AddModelPanel/HuggingFaceFolder/HuggingFaceForm';
import { InstallModelForm } from './AddModelPanel/InstallModelForm';
import { LaunchpadForm } from './AddModelPanel/LaunchpadForm/LaunchpadForm';
import { ModelInstallQueue } from './AddModelPanel/ModelInstallQueue/ModelInstallQueue';
import { ScanModelsForm } from './AddModelPanel/ScanFolder/ScanFolderForm';

const installModelsTabSx: SystemStyleObject = {
  display: 'flex',
  gap: 2,
  px: 2,
};

export const InstallModels = memo(() => {
  const { t } = useTranslation();
  const tabIndex = useStore($installModelsTabIndex);

  {/* TO DO: This click target points to an out-of-date invokeai.ai URL. Reinstate when there is an updated web link. */}
  // const onClickLearnMore = useCallback(() => {
  //   window.open('https://support.invoke.ai/support/solutions/articles/151000170961-supported-models');
  // }, []);

  return (
    <Flex layerStyle="first" borderRadius="base" w="full" h="full" flexDir="column" gap={4}>
      <Flex alignItems="center" justifyContent="space-between">
        <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
        {/* TO DO: This button points to an out-of-date invokeai.ai URL. Reinstate when there is an updated web link. */}
        {/* <Button alignItems="center" variant="link" leftIcon={<PiInfoBold />} onClick={onClickLearnMore}>
          <Text variant="subtext">{t('modelManager.learnMoreAboutSupportedModels')}</Text>
        </Button> */}
      </Flex>
      <Tabs
        variant="line"
        height="100%"
        display="flex"
        flexDir="column"
        index={tabIndex}
        onChange={$installModelsTabIndex.set}
      >
        <TabList>
          <Tab sx={installModelsTabSx}>
            <PiCubeBold />
            {t('modelManager.launchpadTab')}
          </Tab>
          <Tab sx={installModelsTabSx}>
            <PiLinkSimpleBold />
            {t('modelManager.urlOrLocalPath')}
          </Tab>
          <Tab sx={installModelsTabSx}>
            <SiHuggingface />
            {t('modelManager.huggingFace')}
          </Tab>
          <Tab sx={installModelsTabSx}>
            <PiFolderOpenBold />
            {t('modelManager.scanFolder')}
          </Tab>
          <Tab sx={installModelsTabSx}>
            <PiShootingStarBold />
            {t('modelManager.starterModels')}
          </Tab>
        </TabList>
        <TabPanels height="100%">
          <TabPanel height="100%">
            <LaunchpadForm />
          </TabPanel>
          <TabPanel>
            <InstallModelForm />
          </TabPanel>
          <TabPanel height="100%">
            <HuggingFaceForm />
          </TabPanel>
          <TabPanel height="100%">
            <ScanModelsForm />
          </TabPanel>
          <TabPanel height="100%">
            <StarterModelsForm />
          </TabPanel>
        </TabPanels>
      </Tabs>
      <Box layerStyle="second" borderRadius="base" h="50%">
        <ModelInstallQueue />
      </Box>
    </Flex>
  );
});

InstallModels.displayName = 'InstallModels';
