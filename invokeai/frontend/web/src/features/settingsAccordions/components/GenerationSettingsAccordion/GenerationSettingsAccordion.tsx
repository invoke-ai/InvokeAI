import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Expander,
  Flex,
  FormControlGroup,
  StandaloneAccordion,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import { selectLoraSlice } from 'features/lora/store/loraSlice';
import { SyncModelsIconButton } from 'features/modelManagerV2/components/SyncModels/SyncModelsIconButton';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import ParamMainModelSelect from 'features/parameters/components/MainModel/ParamMainModelSelect';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { filter } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

export const GenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const modelConfig = useSelectedModelConfig();
  const selectBadges = useMemo(
    () =>
      createMemoizedSelector(selectLoraSlice, (lora) => {
        const enabledLoRAsCount = filter(lora.loras, (l) => !!l.isEnabled).length;
        const loraTabBadges = enabledLoRAsCount ? [enabledLoRAsCount] : EMPTY_ARRAY;
        const accordionBadges = modelConfig ? [modelConfig.name, modelConfig.base] : EMPTY_ARRAY;
        return { loraTabBadges, accordionBadges };
      }),
    [modelConfig]
  );
  const { loraTabBadges, accordionBadges } = useAppSelector(selectBadges);
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'generation-settings-advanced',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'generation-settings',
    defaultIsOpen: true,
  });

  return (
    <StandaloneAccordion
      label={t('accordions.generation.title')}
      badges={accordionBadges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Tabs variant="collapse">
        <TabList>
          <Tab>{t('accordions.generation.modelTab')}</Tab>
          <Tab badges={loraTabBadges}>{t('accordions.generation.conceptsTab')}</Tab>
        </TabList>
        <TabPanels>
          <TabPanel overflow="visible" px={4} pt={4}>
            <Flex gap={4} alignItems="center">
              <ParamMainModelSelect />
              <Flex>
                <UseDefaultSettingsButton />
                <SyncModelsIconButton />
              </Flex>
            </Flex>
            <Expander isOpen={isOpenExpander} onToggle={onToggleExpander}>
              <Flex gap={4} flexDir="column" pb={4}>
                <FormControlGroup formLabelProps={formLabelProps}>
                  <ParamScheduler />
                  <ParamSteps />
                  <ParamCFGScale />
                </FormControlGroup>
              </Flex>
            </Expander>
          </TabPanel>
          <TabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <LoRASelect />
              <LoRAList />
            </Flex>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </StandaloneAccordion>
  );
});

GenerationSettingsAccordion.displayName = 'GenerationSettingsAccordion';
