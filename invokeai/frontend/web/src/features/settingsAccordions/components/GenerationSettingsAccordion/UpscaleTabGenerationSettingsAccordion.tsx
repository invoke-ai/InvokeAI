import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Box, Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import ParamGuidance from 'features/parameters/components/Core/ParamGuidance';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import { DisabledModelWarning } from 'features/parameters/components/MainModel/DisabledModelWarning';
import ParamUpscaleCFGScale from 'features/parameters/components/Upscale/ParamUpscaleCFGScale';
import ParamUpscaleScheduler from 'features/parameters/components/Upscale/ParamUpscaleScheduler';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';
import { API_BASE_MODELS } from 'features/parameters/types/constants';
import { MainModelPicker } from 'features/settingsAccordions/components/GenerationSettingsAccordion/MainModelPicker';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { isFluxFillMainModelModelConfig } from 'services/api/types';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

export const UpscaleTabGenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const modelConfig = useSelectedModelConfig();
  const isFLUX = useAppSelector(selectIsFLUX);

  const isApiModel = useIsApiModel();

  const selectBadges = useMemo(
    () =>
      createMemoizedSelector(selectLoRAsSlice, (loras) => {
        const enabledLoRAsCount = loras.loras.filter((l) => l.isEnabled).length;
        const loraTabBadges = enabledLoRAsCount ? [`${enabledLoRAsCount} ${t('models.concepts')}`] : EMPTY_ARRAY;
        const accordionBadges =
          modelConfig && API_BASE_MODELS.includes(modelConfig.base)
            ? [modelConfig.name]
            : modelConfig
              ? [modelConfig.name, modelConfig.base]
              : EMPTY_ARRAY;
        return { loraTabBadges, accordionBadges };
      }),
    [modelConfig, t]
  );
  const { loraTabBadges, accordionBadges } = useAppSelector(selectBadges);
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'generation-settings-advanced',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: `generation-settings-upscaling`,
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion
      label={t('accordions.generation.title')}
      badges={[...accordionBadges, ...loraTabBadges]}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Box px={4} pt={4} data-testid="generation-accordion">
        <Flex gap={4} flexDir="column" pb={isApiModel ? 4 : 0}>
          <DisabledModelWarning />
          <MainModelPicker />
          {!isApiModel && <LoRASelect />}
          {!isApiModel && <LoRAList />}
        </Flex>
        {!isApiModel && (
          <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
            <Flex gap={4} flexDir="column" pb={4}>
              <FormControlGroup formLabelProps={formLabelProps}>
                <ParamUpscaleScheduler />
                <ParamSteps />
                {isFLUX && modelConfig && !isFluxFillMainModelModelConfig(modelConfig) && <ParamGuidance />}
                <ParamUpscaleCFGScale />
              </FormControlGroup>
            </Flex>
          </Expander>
        )}
      </Box>
    </StandaloneAccordion>
  );
});

UpscaleTabGenerationSettingsAccordion.displayName = 'UpscaleTabGenerationSettingsAccordion';
