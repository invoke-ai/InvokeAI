import { Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StarterModelsForm } from 'features/modelManagerV2/subpanels/AddModelPanel/StarterModels/StarterModelsForm';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Panel, PanelGroup } from 'react-resizable-panels';

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
    <PanelGroup direction="vertical">
      <Panel>
        <Flex w="full" h="full" flexDir="column" gap={2}>
          <Heading fontSize="xl">{t('modelManager.addModel')}</Heading>
          <Tabs variant="collapse" height="full" display="flex" flexDir="column" index={index} onChange={onChange}>
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
        </Flex>
      </Panel>
      <ResizeHandle orientation="horizontal" />
      <Panel>
        {/* <Box layerStyle="second" borderRadius="base" h="full"> */}
          <ModelInstallQueue />
        {/* </Box> */}
      </Panel>
    </PanelGroup>
  );
});

InstallModels.displayName = 'InstallModels';
