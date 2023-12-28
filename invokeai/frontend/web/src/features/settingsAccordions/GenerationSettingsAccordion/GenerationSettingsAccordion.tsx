import { Flex } from '@chakra-ui/layout';
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
import { useTranslation } from 'react-i18next';

const labelProps: InvLabelProps = {
  w: '4rem',
};

export const GenerationSettingsAccordion = () => {
  const { t } = useTranslation();
  const loraCount = useAppSelector((state) => size(state.lora.loras));

  return (
    <InvSingleAccordion
      label={t('accordions.generation.title')}
      defaultIsOpen={true}
    >
      <InvTabs variant="collapse">
        <InvTabList>
          <InvTab>{t('accordions.generation.modelTab')}</InvTab>
          <InvTab badges={loraCount ? [loraCount] : undefined}>
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
};
