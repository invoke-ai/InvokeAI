import { Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { GenerateLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/GenerateLaunchpadPanel';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer2';
import { ProgressImage } from 'features/gallery/components/ImageViewer/ProgressImage2';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar2';
import { memo } from 'react';

export const SimpleSession = memo(() => {
  return (
    <Tabs w="full" h="full" px={2}>
      <TabList>
        <Tab>Launchpad</Tab>
        <Tab>Viewer</Tab>
        <Tab>Generation Progress</Tab>
      </TabList>
      <TabPanels w="full" h="full">
        <TabPanel w="full" h="full" justifyContent="center">
          <GenerateLaunchpadPanel />
        </TabPanel>
        <TabPanel w="full" h="full">
          <Flex flexDir="column" w="full" h="full">
            <ViewerToolbar />
            <ImageViewer />
          </Flex>
        </TabPanel>
        <TabPanel w="full" h="full">
          <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2}>
            <ProgressImage />
          </Flex>
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
});
SimpleSession.displayName = 'SimpleSession';
