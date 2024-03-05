import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

import { InstallModelForm } from './AddModelPanel/InstallModelForm';
import { ModelInstallQueue } from './AddModelPanel/ModelInstallQueue/ModelInstallQueue';
import { ScanModelsForm } from './AddModelPanel/ScanFolder/ScanFolderForm';

export const InstallModels = () => {
  const { t } = useTranslation();
  return (
    <Flex layerStyle="first" p={3} borderRadius="base" w="full" h="full" flexDir="column" gap={2}>
      <Box w="full" p={2}>
        <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="50%" overflow="hidden">
        <Tabs variant="collapse" height="100%">
          <TabList>
            <Tab>{t('common.simple')}</Tab>
            <Tab>{t('modelManager.scan')}</Tab>
          </TabList>
          <TabPanels p={3} height="100%">
            <TabPanel>
              <InstallModelForm />
            </TabPanel>
            <TabPanel height="100%">
              <ScanModelsForm />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="50%">
        <ModelInstallQueue />
      </Box>
    </Flex>
  );
};
