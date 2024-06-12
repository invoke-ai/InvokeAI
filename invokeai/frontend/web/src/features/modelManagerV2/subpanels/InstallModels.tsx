import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { StarterModelsForm } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterModelsForm';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useMainModels } from 'services/api/hooks/modelsByType';

import { HuggingFaceForm } from './AddModelPanel/HuggingFaceFolder/HuggingFaceForm';
import { InstallModelForm } from './AddModelPanel/InstallModelForm';
import { ModelInstallQueue } from './AddModelPanel/ModelInstallQueue/ModelInstallQueue';
import { ScanModelsForm } from './AddModelPanel/ScanFolder/ScanFolderForm';

export const InstallModels = () => {
  const { t } = useTranslation();
  const [mainModels, { data }] = useMainModels();
  const downloadHFModel = useAppSelector((state) => state.modelmanagerV2.downloadHFModel);

  const [selectedTabIndex, setSelectedTabIndex] = useState(data && mainModels.length ? 0 : 3);

  useEffect(() => {
    if (downloadHFModel) {
      setSelectedTabIndex(1);
    }
  }, [downloadHFModel]);

  return (
    <Flex layerStyle="first" borderRadius="base" w="full" h="full" flexDir="column" gap={4}>
      <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
      <Tabs
        variant="collapse"
        height="50%"
        display="flex"
        flexDir="column"
        index={selectedTabIndex}
        onChange={setSelectedTabIndex}
      >
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
};
