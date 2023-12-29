import { Box, Flex } from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import QueueControls from 'features/queue/components/QueueControls';
import { SDXLPrompts } from 'features/sdxl/components/SDXLPrompts/SDXLPrompts';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { ControlSettingsAccordion } from 'features/settingsAccordions/ControlSettingsAccordion/ControlSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { ImageSettingsAccordion } from 'features/settingsAccordions/ImageSettingsAccordion/ImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo } from 'react';

const ParametersPanel = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isSDXL = useAppSelector(
    (state: RootState) => state.generation.model?.base_model === 'sdxl'
  );

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <QueueControls />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          <OverlayScrollbarsComponent
            defer
            style={{ height: '100%', width: '100%' }}
            options={overlayScrollbarsParams.options}
          >
            <Flex gap={2} flexDirection="column" h="full" w="full">
              {isSDXL ? <SDXLPrompts /> : <Prompts />}
              <ImageSettingsAccordion />
              <GenerationSettingsAccordion />
              <ControlSettingsAccordion />
              {activeTabName === 'unifiedCanvas' && (
                <CompositingSettingsAccordion />
              )}
              {isSDXL && <RefinerSettingsAccordion />}
              <AdvancedSettingsAccordion />
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(ParametersPanel);
