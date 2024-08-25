import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StarterModelsForm } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterModelsForm';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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

  return (
    <Flex layerStyle="body" w="full" h="full" flexDir="column" gap={4}>
      <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
      <Tabs borderRadius="base" bg="base.850" size="lg" isFitted variant="line" height="50%" display="flex" flexDir="column" index={index} onChange={onChange}>
        <TabList>
          <Tab py={2}>{t('modelManager.urlOrLocalPath')}</Tab>
          <Tab py={2}>{t('modelManager.huggingFace')}</Tab>
          <Tab py={2}>{t('modelManager.scanFolder')}</Tab>
          <Tab py={2}>{t('modelManager.starterModels')}</Tab>
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
      <Box layerStyle="first" borderRadius="base" h="50%">
        <ModelInstallQueue />
      </Box>
    </Flex>
  );
});

InstallModels.displayName = 'InstallModels';
