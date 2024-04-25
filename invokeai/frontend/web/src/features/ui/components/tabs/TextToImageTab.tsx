import { Box, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { RegionalPromptsEditor } from 'features/regionalPrompts/components/RegionalPromptsEditor';
import { useRegionalControlTitle } from 'features/regionalPrompts/hooks/useRegionalControlTitle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const TextToImageTab = () => {
  const { t } = useTranslation();
  const regionalControlTitle = useRegionalControlTitle();

  return (
    <Box position="relative" w="full" h="full" p={2} borderRadius="base">
      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('common.viewer')}</Tab>
          <Tab>{regionalControlTitle}</Tab>
        </TabList>

        <TabPanels w="full" h="full" minH={0} minW={0}>
          <TabPanel>
            <CurrentImageDisplay />
          </TabPanel>
          <TabPanel>
            <RegionalPromptsEditor />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default memo(TextToImageTab);
