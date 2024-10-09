import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Box, Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamGuidance from 'features/parameters/components/Core/ParamGuidance';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import ParamMainModelSelect from 'features/parameters/components/MainModel/ParamMainModelSelect';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

export const GenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const modelConfig = useSelectedModelConfig();
  const activeTabName = useAppSelector(selectActiveTab);
  const isFLUX = useAppSelector(selectIsFLUX);
  const selectBadges = useMemo(
    () =>
      createMemoizedSelector(selectLoRAsSlice, (loras) => {
        const enabledLoRAsCount = loras.loras.filter((l) => l.isEnabled).length;
        const loraTabBadges = enabledLoRAsCount ? [`${enabledLoRAsCount} ${t('models.concepts')}`] : EMPTY_ARRAY;
        const accordionBadges = modelConfig ? [modelConfig.name, modelConfig.base] : EMPTY_ARRAY;
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
    id: `generation-settings-${activeTabName}`,
    defaultIsOpen: activeTabName !== 'upscaling',
  });

  return (
    <StandaloneAccordion
      label={t('accordions.generation.title')}
      badges={[...accordionBadges, ...loraTabBadges]}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Box px={4} pt={4} data-testid="generation-accordion">
        <Flex gap={4} flexDir="column">
          <Flex gap={4} alignItems="center">
            <ParamMainModelSelect />
            <Flex>
              <UseDefaultSettingsButton />
              <NavigateToModelManagerButton />
            </Flex>
          </Flex>
          <Flex gap={4} flexDir="column">
            <LoRASelect />
            <LoRAList />
          </Flex>
        </Flex>
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} flexDir="column" pb={4}>
            <FormControlGroup formLabelProps={formLabelProps}>
              {!isFLUX && <ParamScheduler />}
              <ParamSteps />
              {isFLUX ? <ParamGuidance /> : <ParamCFGScale />}
            </FormControlGroup>
          </Flex>
        </Expander>
      </Box>
    </StandaloneAccordion>
  );
});

GenerationSettingsAccordion.displayName = 'GenerationSettingsAccordion';
