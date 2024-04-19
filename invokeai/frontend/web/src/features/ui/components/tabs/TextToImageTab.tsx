import { Box, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { RegionalPromptsEditor } from 'features/regionalPrompts/components/RegionalPromptsEditor';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const TextToImageTab = () => {
  const { t } = useTranslation();
  const noOfRPLayers = useAppSelector(
    (s) => s.regionalPrompts.present.layers.filter((l) => l.kind === 'regionalPromptLayer' && l.isVisible).length
  );
  return (
    <Box position="relative" w="full" h="full" p={2} borderRadius="base">
      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('common.viewer')}</Tab>
          <Tab>
            {t('regionalPrompts.regionalPrompts')}
            {noOfRPLayers > 0 ? ` (${noOfRPLayers})` : ''}
          </Tab>
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
