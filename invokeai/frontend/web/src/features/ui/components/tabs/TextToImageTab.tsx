import { Box, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { ControlLayersEditor } from 'features/controlLayers/components/ControlLayersEditor';
import { useControlLayersTitle } from 'features/controlLayers/hooks/useControlLayersTitle';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const TextToImageTab = () => {
  const { t } = useTranslation();
  const controlLayersTitle = useControlLayersTitle();

  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('common.viewer')}</Tab>
          <Tab>{controlLayersTitle}</Tab>
        </TabList>

        <TabPanels w="full" h="full" minH={0} minW={0}>
          <TabPanel>
            <CurrentImageDisplay />
          </TabPanel>
          <TabPanel>
            <ControlLayersEditor />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default memo(TextToImageTab);
