import { Box, Button, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/components/ModelCombobox/ModelCombobox';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { selectIsCogView4, selectIsSDXL } from 'features/controlLayers/store/paramsSlice';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/components/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { ImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { StylePresetMenu } from 'features/stylePresets/components/StylePresetMenu';
import { StylePresetMenuTrigger } from 'features/stylePresets/components/StylePresetMenuTrigger';
import { $isStylePresetsMenuOpen } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import { useAllModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const onSelect = (modelConfig: AnyModelConfig) => {
  // Handle model selection
  toast({
    description: `Selected model: ${modelConfig.name}`,
  });
};

const ParametersPanelTextToImage = () => {
  const isSDXL = useAppSelector(selectIsSDXL);
  const isCogview4 = useAppSelector(selectIsCogView4);
  const isStylePresetsMenuOpen = useStore($isStylePresetsMenuOpen);
  const [modelConfigs] = useAllModels();
  const modelCmdk = useModelCombobox({ onSelect, modelConfigs });

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
              <Button onClick={modelCmdk.onOpen}>model</Button>
              <Prompts />
              <ImageSettingsAccordion />
              <GenerationSettingsAccordion />
              <CompositingSettingsAccordion />
              {isSDXL && <RefinerSettingsAccordion />}
              {!isCogview4 && <AdvancedSettingsAccordion />}
            </Flex>
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(ParametersPanelTextToImage);
