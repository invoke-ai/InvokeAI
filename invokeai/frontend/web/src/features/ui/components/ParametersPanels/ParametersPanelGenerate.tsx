import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { selectIsCogView4, selectIsSDXL } from 'features/controlLayers/store/paramsSlice';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { GenerateTabImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/GenerateTabImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
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

export const ParametersPanelGenerate = memo(() => {
  const isSDXL = useAppSelector(selectIsSDXL);
  const isCogview4 = useAppSelector(selectIsCogView4);
  const isStylePresetsMenuOpen = useStore($isStylePresetsMenuOpen);

  const isApiModel = useIsApiModel();

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
              <GenerateTabImageSettingsAccordion />
              <GenerationSettingsAccordion />
              {isSDXL && <RefinerSettingsAccordion />}
              {!isCogview4 && !isApiModel && <AdvancedSettingsAccordion />}
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
});

ParametersPanelGenerate.displayName = 'ParametersPanelGenerate';
