import { Box, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { RegionalPromptsEditor } from 'features/regionalPrompts/components/RegionalPromptsEditor';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectValidLayerCount = createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  if (!regionalPrompts.present.isEnabled) {
    return 0;
  }
  const validLayers = regionalPrompts.present.layers
    .filter((l) => l.isVisible)
    .filter((l) => {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasAtLeastOneImagePrompt = l.ipAdapterIds.length > 0;
      return hasTextPrompt || hasAtLeastOneImagePrompt;
    });

  return validLayers.length;
});

const TextToImageTab = () => {
  const { t } = useTranslation();
  const validLayerCount = useAppSelector(selectValidLayerCount);
  return (
    <Box position="relative" w="full" h="full" p={2} borderRadius="base">
      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('common.viewer')}</Tab>
          <Tab>
            {t('regionalPrompts.regionalPrompts')}
            {validLayerCount > 0 ? ` (${validLayerCount})` : ''}
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
