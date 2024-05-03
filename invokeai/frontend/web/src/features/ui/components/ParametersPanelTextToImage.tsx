import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { ControlLayersPanelContent } from 'features/controlLayers/components/ControlLayersPanelContent';
import { useControlLayersTitle } from 'features/controlLayers/hooks/useControlLayersTitle';
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

const unselectedStyles: ChakraProps['sx'] = {
  bg: 'none',
  color: 'base.300',
  fontWeight: 'semibold',
  fontSize: 'sm',
  w: '50%',
  borderWidth: 1,
  borderRadius: 'base',
};

const selectedStyles: ChakraProps['sx'] = {
  color: 'invokeBlue.300',
  borderColor: 'invokeBlueAlpha.400',
  _hover: {
    color: 'invokeBlue.200',
  },
};

const hoverStyles: ChakraProps['sx'] = {
  bg: 'base.850',
  color: 'base.100',
};

const ParametersPanelTextToImage = () => {
  const { t } = useTranslation();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const controlLayersTitle = useControlLayersTitle();
  const isSDXL = useAppSelector((s) => s.generation.model?.base === 'sdxl');

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <QueueControls />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            <Flex gap={2} flexDirection="column" h="full" w="full">
              {isSDXL ? <SDXLPrompts /> : <Prompts />}
              <Tabs variant="unstyled" display="flex" flexDir="column" w="full" h="full" gap={2}>
                <TabList gap={2}>
                  <Tab _hover={hoverStyles} _selected={selectedStyles} sx={unselectedStyles}>
                    {t('common.settingsLabel')}
                  </Tab>
                  <Tab _hover={hoverStyles} _selected={selectedStyles} sx={unselectedStyles}>
                    {controlLayersTitle}
                  </Tab>
                </TabList>
                <TabPanels w="full" h="full">
                  <TabPanel p={0}>
                    <Flex gap={2} flexDirection="column" h="full" w="full">
                      <ImageSettingsAccordion />
                      <GenerationSettingsAccordion />
                      {activeTabName !== 'generation' && <ControlSettingsAccordion />}
                      {activeTabName === 'canvas' && <CompositingSettingsAccordion />}
                      {isSDXL && <RefinerSettingsAccordion />}
                      <AdvancedSettingsAccordion />
                    </Flex>
                  </TabPanel>
                  <TabPanel p={0}>
                    <ControlLayersPanelContent />
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
