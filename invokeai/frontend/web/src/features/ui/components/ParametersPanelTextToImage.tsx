import { Box, Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { RegionalPromptsPanelContent } from 'features/controlLayers/components/RegionalPromptsPanelContent';
import { useRegionalControlTitle } from 'features/controlLayers/hooks/useRegionalControlTitle';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import QueueControls from 'features/queue/components/QueueControls';
import { SDXLPrompts } from 'features/sdxl/components/SDXLPrompts/SDXLPrompts';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/components/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { ControlSettingsAccordion } from 'features/settingsAccordions/components/ControlSettingsAccordion/ControlSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { ImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const ParametersPanelTextToImage = () => {
  const { t } = useTranslation();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const regionalControlTitle = useRegionalControlTitle();
  const isSDXL = useAppSelector((s) => s.generation.model?.base === 'sdxl');

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <QueueControls />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            <Flex gap={2} flexDirection="column" h="full" w="full">
              {isSDXL ? <SDXLPrompts /> : <Prompts />}
              <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
                <TabList>
                  <Tab>{t('parameters.globalSettings')}</Tab>
                  <Tab>{regionalControlTitle}</Tab>
                </TabList>

                <TabPanels w="full" h="full">
                  <TabPanel>
                    <Flex gap={2} flexDirection="column" h="full" w="full">
                      <ImageSettingsAccordion />
                      <GenerationSettingsAccordion />
                      {activeTabName !== 'txt2img' && <ControlSettingsAccordion />}
                      {activeTabName === 'unifiedCanvas' && <CompositingSettingsAccordion />}
                      {isSDXL && <RefinerSettingsAccordion />}
                      <AdvancedSettingsAccordion />
                    </Flex>
                  </TabPanel>
                  <TabPanel>
                    <RegionalPromptsPanelContent />
                  </TabPanel>
                </TabPanels>
              </Tabs>
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(ParametersPanelTextToImage);
