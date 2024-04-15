import { Box, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { RegionalPromptsEditor } from 'features/regionalPrompts/components/RegionalPromptsEditor';
import { memo } from 'react';

const TextToImageTab = () => {
  return (
    <Box position="relative" w="full" h="full" p={2} borderRadius="base">
      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>Viewer</Tab>
          <Tab>Regional Prompts</Tab>
        </TabList>

        <TabPanels w="full" h="full">
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
