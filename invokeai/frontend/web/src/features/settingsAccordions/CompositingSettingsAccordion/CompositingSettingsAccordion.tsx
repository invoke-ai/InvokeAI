import { Flex } from '@chakra-ui/layout';
import type { FormLabelProps } from '@chakra-ui/react';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import { InvTab } from 'common/components/InvTabs/InvTab';
import {
  InvTabList,
  InvTabPanel,
  InvTabPanels,
  InvTabs,
} from 'common/components/InvTabs/wrapper';
import ParamCanvasCoherenceMode from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceMode';
import ParamCanvasCoherenceSteps from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceSteps';
import ParamCanvasCoherenceStrength from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceStrength';
import ParamMaskBlur from 'features/parameters/components/Canvas/Compositing/MaskAdjustment/ParamMaskBlur';
import ParamMaskBlurMethod from 'features/parameters/components/Canvas/Compositing/MaskAdjustment/ParamMaskBlurMethod';
import ParamInfillMethod from 'features/parameters/components/Canvas/InfillAndScaling/ParamInfillMethod';
import ParamInfillOptions from 'features/parameters/components/Canvas/InfillAndScaling/ParamInfillOptions';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const coherenceLabelProps: FormLabelProps = {
  w: '4.5rem',
};

export const CompositingSettingsAccordion = memo(() => {
  const { t } = useTranslation();

  return (
    <InvSingleAccordion label={t('accordions.compositing.title')}>
      <InvTabs variant="collapse">
        <InvTabList>
          <InvTab>{t('accordions.compositing.coherenceTab')}</InvTab>
          <InvTab>{t('accordions.compositing.infillTab')}</InvTab>
        </InvTabList>
        <InvTabPanels>
          <InvTabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <InvControlGroup labelProps={coherenceLabelProps}>
                <ParamCanvasCoherenceMode />
                <ParamCanvasCoherenceSteps />
                <ParamCanvasCoherenceStrength />
                <ParamMaskBlurMethod />
                <ParamMaskBlur />
              </InvControlGroup>
            </Flex>
          </InvTabPanel>
          <InvTabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <InvControlGroup labelProps={coherenceLabelProps}>
                <ParamInfillMethod />
                <ParamInfillOptions />
              </InvControlGroup>
            </Flex>
          </InvTabPanel>
        </InvTabPanels>
      </InvTabs>
    </InvSingleAccordion>
  );
});

CompositingSettingsAccordion.displayName = 'CompositingSettingsAccordion';
