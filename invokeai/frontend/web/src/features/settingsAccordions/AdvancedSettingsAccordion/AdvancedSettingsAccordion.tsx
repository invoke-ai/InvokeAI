import { Flex } from '@chakra-ui/layout';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import ParamCFGRescaleMultiplier from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import ParamClipSkip from 'features/parameters/components/Advanced/ParamClipSkip';
import ParamSeamlessXAxis from 'features/parameters/components/Seamless/ParamSeamlessXAxis';
import ParamSeamlessYAxis from 'features/parameters/components/Seamless/ParamSeamlessYAxis';
import ParamVAEModelSelect from 'features/parameters/components/VAEModel/ParamVAEModelSelect';
import ParamVAEPrecision from 'features/parameters/components/VAEModel/ParamVAEPrecision';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const labelProps: InvLabelProps = {
  minW: '9.2rem',
};

const labelProps2: InvLabelProps = {
  flexGrow: 1,
};

export const AdvancedSettingsAccordion = memo(() => {
  const { t } = useTranslation();

  return (
    <InvSingleAccordion label={t('accordions.advanced.title')}>
      <Flex gap={4} alignItems="center" p={4} flexDir="column">
        <Flex gap={4} w="full">
          <ParamVAEModelSelect />
          <ParamVAEPrecision />
        </Flex>
        <InvControlGroup labelProps={labelProps}>
          <ParamClipSkip />
          <ParamCFGRescaleMultiplier />
        </InvControlGroup>
        <Flex gap={4} w="full">
          <InvControlGroup labelProps={labelProps2}>
            <ParamSeamlessXAxis />
            <ParamSeamlessYAxis />
          </InvControlGroup>
        </Flex>
      </Flex>
    </InvSingleAccordion>
  );
});

AdvancedSettingsAccordion.displayName = 'AdvancedSettingsAccordion';
