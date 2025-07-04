import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import { UpscaleTabAdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/UpscaleTabAdvancedSettingsAccordion';
import { UpscaleTabGenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/UpscaleTabGenerationSettingsAccordion';
import { UpscaleSettingsAccordion } from 'features/settingsAccordions/components/UpscaleSettingsAccordion/UpscaleSettingsAccordion';
import { StylePresetMenu } from 'features/stylePresets/components/StylePresetMenu';
import { StylePresetMenuTrigger } from 'features/stylePresets/components/StylePresetMenuTrigger';
import { $isStylePresetsMenuOpen } from 'features/stylePresets/store/stylePresetSlice';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

export const ParametersPanelUpscale = memo(() => {
  const isStylePresetsMenuOpen = useStore($isStylePresetsMenuOpen);

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <StylePresetMenuTrigger />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          {isStylePresetsMenuOpen && (
            <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
              <Flex gap={2} flexDirection="column" h="full" w="full">
                <StylePresetMenu />
              </Flex>
            </OverlayScrollbarsComponent>
          )}
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            <Flex gap={2} flexDirection="column" h="full" w="full">
              <Prompts />
              <UpscaleSettingsAccordion />
              <UpscaleTabGenerationSettingsAccordion />
              <UpscaleTabAdvancedSettingsAccordion />
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
});

ParametersPanelUpscale.displayName = 'ParametersPanelUpscale';
