import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

import { AdvancedImport } from './AddModelPanel/AdvancedImport';
import { ImportQueue } from './AddModelPanel/ImportQueue/ImportQueue';
import { ScanModelsForm } from './AddModelPanel/ScanModels/ScanModelsForm';
import { SimpleImport } from './AddModelPanel/SimpleImport';

export const ImportModels = () => {
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
            <Tab>{t('modelManager.advanced')}</Tab>
            <Tab>{t('modelManager.scan')}</Tab>
          </TabList>
          <TabPanels p={3} height="100%">
            <TabPanel>
              <SimpleImport />
            </TabPanel>
            <TabPanel height="100%">
              <AdvancedImport />
            </TabPanel>
            <TabPanel height="100%">
              <ScanModelsForm />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="50%">
        <ImportQueue />
      </Box>
    </Flex>
  );
};
