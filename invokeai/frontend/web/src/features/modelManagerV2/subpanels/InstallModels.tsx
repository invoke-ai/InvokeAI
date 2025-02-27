import { Box, Button, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StarterModelsForm } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterModelsForm';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

import { HuggingFaceForm } from './AddModelPanel/HuggingFaceFolder/HuggingFaceForm';
import { InstallModelForm } from './AddModelPanel/InstallModelForm';
import { ModelInstallQueue } from './AddModelPanel/ModelInstallQueue/ModelInstallQueue';
import { ScanModelsForm } from './AddModelPanel/ScanFolder/ScanFolderForm';

export const $installModelsTab = atom(0);

export const InstallModels = memo(() => {
  const { t } = useTranslation();
  const index = useStore($installModelsTab);
  const onChange = useCallback((index: number) => {
    $installModelsTab.set(index);
  }, []);

  const onClickLearnMore = useCallback(() => {
    window.open('https://support.invoke.ai/support/solutions/articles/151000170961-supported-models');
  }, []);

  return (
    <Flex layerStyle="first" borderRadius="base" w="full" h="full" flexDir="column" gap={4}>
      <Flex alignItems="center" justifyContent="space-between">
        <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
        <Button alignItems="center" variant="link" leftIcon={<PiInfoBold />} onClick={onClickLearnMore}>
          <Text variant="subtext">{t('modelManager.learnMoreAboutSupportedModels')}</Text>
        </Button>
      </Flex>

      <Tabs variant="collapse" height="50%" display="flex" flexDir="column" index={index} onChange={onChange}>
        <TabList>
          <Tab>{t('modelManager.urlOrLocalPath')}</Tab>
          <Tab>{t('modelManager.huggingFace')}</Tab>
          <Tab>{t('modelManager.scanFolder')}</Tab>
          <Tab>{t('modelManager.starterModels')}</Tab>
        </TabList>
        <TabPanels p={3} height="100%">
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
