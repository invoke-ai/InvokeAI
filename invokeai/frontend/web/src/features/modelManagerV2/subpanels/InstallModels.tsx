import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

import { HuggingFaceForm } from './AddModelPanel/HuggingFaceFolder/HuggingFaceForm';
import { InstallModelForm } from './AddModelPanel/InstallModelForm';
import { ModelInstallQueue } from './AddModelPanel/ModelInstallQueue/ModelInstallQueue';
import { ScanModelsForm } from './AddModelPanel/ScanFolder/ScanFolderForm';

export const InstallModels = () => {
  const { t } = useTranslation();
  return (
    <Flex layerStyle="first" borderRadius="base" w="full" h="full" flexDir="column" gap={4}>
      <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
      <Tabs variant="collapse" height="50%" display="flex" flexDir="column">
        <TabList>
          <Tab>{t('common.simple')}</Tab>
          <Tab>{t('modelManager.huggingFace')}</Tab>
          <Tab>{t('modelManager.scan')}</Tab>
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
        </TabPanels>
      </Tabs>
      <Box layerStyle="second" borderRadius="base" h="50%">
        <ModelInstallQueue />
      </Box>
    </Flex>
  );
};
