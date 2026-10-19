import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Divider, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { memo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiLinkSimpleBold } from 'react-icons/pi';

import { CustomNodesInstallLog } from './CustomNodesInstallLog';
import { InstallFromGitForm } from './InstallFromGitForm';
import { ScanNodesForm } from './ScanNodesForm';

const paneSx: SystemStyleObject = {
  layerStyle: 'first',
  p: 4,
  borderRadius: 'base',
  w: {
    base: '50%',
    lg: '75%',
    '2xl': '85%',
  },
  h: 'full',
  minWidth: '300px',
  overflow: 'auto',
};

const installTabSx: SystemStyleObject = {
  display: 'flex',
  gap: 2,
  px: 2,
};

export const CustomNodesInstallPane = memo(() => {
  const { t } = useTranslation();
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <Flex sx={paneSx} flexDir="column" gap={4}>
      <Heading fontSize="xl">{t('customNodes.installTitle')}</Heading>
      <Tabs variant="line" height="100%" display="flex" flexDir="column" index={tabIndex} onChange={setTabIndex}>
        <TabList>
          <Tab sx={installTabSx}>
            <PiLinkSimpleBold />
            {t('customNodes.gitUrl')}
          </Tab>
          <Tab sx={installTabSx}>
            <PiFolderOpenBold />
            {t('customNodes.scanFolder')}
          </Tab>
        </TabList>
        <TabPanels height="100%">
          <TabPanel>
            <InstallFromGitForm />
          </TabPanel>
          <TabPanel height="100%">
            <ScanNodesForm />
          </TabPanel>
        </TabPanels>
      </Tabs>
      <Divider />
      <Box h="50%">
        <CustomNodesInstallLog />
      </Box>
    </Flex>
  );
});

CustomNodesInstallPane.displayName = 'CustomNodesInstallPane';
