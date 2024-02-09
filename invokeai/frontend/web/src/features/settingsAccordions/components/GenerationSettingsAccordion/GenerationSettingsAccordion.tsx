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
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import { selectLoraSlice } from 'features/lora/store/loraSlice';
import { SyncModelsIconButton } from 'features/modelManager/components/SyncModels/SyncModelsIconButton';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import ParamMainModelSelect from 'features/parameters/components/MainModel/ParamMainModelSelect';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { filter } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

const badgesSelector = createMemoizedSelector(selectLoraSlice, selectGenerationSlice, (lora, generation) => {
  const enabledLoRAsCount = filter(lora.loras, (l) => !!l.isEnabled).length;
  const loraTabBadges = enabledLoRAsCount ? [enabledLoRAsCount] : [];
  const accordionBadges: (string | number)[] = [];
  if (generation.model) {
    accordionBadges.push(generation.model.model_name);
    accordionBadges.push(generation.model.base_model);
  }

  return { loraTabBadges, accordionBadges };
});

export const GenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { loraTabBadges, accordionBadges } = useAppSelector(badgesSelector);
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
              <SyncModelsIconButton />
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
