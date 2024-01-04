import { Flex } from '@chakra-ui/layout';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvExpander } from 'common/components/InvExpander/InvExpander';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import { InvTab } from 'common/components/InvTabs/InvTab';
import {
  InvTabList,
  InvTabPanel,
  InvTabPanels,
  InvTabs,
} from 'common/components/InvTabs/wrapper';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import { SyncModelsIconButton } from 'features/modelManager/components/SyncModels/SyncModelsIconButton';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import ParamMainModelSelect from 'features/parameters/components/MainModel/ParamMainModelSelect';
import { size } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const labelProps: InvLabelProps = {
  minW: '4rem',
};

const badgesSelector = createMemoizedSelector(
  stateSelector,
  ({ lora, generation }) => {
    const loraTabBadges = size(lora.loras) ? [size(lora.loras)] : [];
    const accordionBadges: (string | number)[] = [];
    if (generation.model) {
      accordionBadges.push(generation.model.model_name);
      accordionBadges.push(generation.model.base_model);
    }

    return { loraTabBadges, accordionBadges };
  }
);

export const GenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { loraTabBadges, accordionBadges } = useAppSelector(badgesSelector);

  return (
    <InvSingleAccordion
      label={t('accordions.generation.title')}
      defaultIsOpen={true}
      badges={accordionBadges}
    >
      <InvTabs variant="collapse">
        <InvTabList>
          <InvTab>{t('accordions.generation.modelTab')}</InvTab>
          <InvTab badges={loraTabBadges}>
            {t('accordions.generation.conceptsTab')}
          </InvTab>
        </InvTabList>
        <InvTabPanels>
          <InvTabPanel overflow="visible" px={4} pt={4}>
            <Flex gap={4}>
              <ParamMainModelSelect />
              <SyncModelsIconButton />
            </Flex>
            <InvExpander>
              <Flex gap={4} flexDir="column" pb={4}>
                <InvControlGroup labelProps={labelProps}>
                  <ParamScheduler />
                  <ParamSteps />
                  <ParamCFGScale />
                </InvControlGroup>
              </Flex>
            </InvExpander>
          </InvTabPanel>
          <InvTabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <LoRASelect />
              <LoRAList />
            </Flex>
          </InvTabPanel>
        </InvTabPanels>
      </InvTabs>
    </InvSingleAccordion>
  );
});

GenerationSettingsAccordion.displayName = 'GenerationSettingsAccordion';
