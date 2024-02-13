import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ParamCFGRescaleMultiplier from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import ParamClipSkip from 'features/parameters/components/Advanced/ParamClipSkip';
import ParamSeamlessXAxis from 'features/parameters/components/Seamless/ParamSeamlessXAxis';
import ParamSeamlessYAxis from 'features/parameters/components/Seamless/ParamSeamlessYAxis';
import ParamVAEModelSelect from 'features/parameters/components/VAEModel/ParamVAEModelSelect';
import ParamVAEPrecision from 'features/parameters/components/VAEModel/ParamVAEPrecision';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const formLabelProps: FormLabelProps = {
  minW: '9.2rem',
};

const formLabelProps2: FormLabelProps = {
  flexGrow: 1,
};

const selectBadges = createMemoizedSelector(selectGenerationSlice, (generation) => {
  const badges: (string | number)[] = [];
  if (generation.vae) {
    let vaeBadge = generation.vae.model_name;
    if (generation.vaePrecision === 'fp16') {
      vaeBadge += ` ${generation.vaePrecision}`;
    }
    badges.push(vaeBadge);
  } else if (generation.vaePrecision === 'fp16') {
    badges.push(`VAE ${generation.vaePrecision}`);
  }
  if (generation.clipSkip) {
    badges.push(`Skip ${generation.clipSkip}`);
  }
  if (generation.cfgRescaleMultiplier) {
    badges.push(`Rescale ${generation.cfgRescaleMultiplier}`);
  }
  if (generation.seamlessXAxis || generation.seamlessYAxis) {
    badges.push('seamless');
  }
  return badges;
});

export const AdvancedSettingsAccordion = memo(() => {
  const badges = useAppSelector(selectBadges);
  const { t } = useTranslation();
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'advanced-settings',
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion label={t('accordions.advanced.title')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex gap={4} alignItems="center" p={4} flexDir="column">
        <Flex gap={4} w="full">
          <ParamVAEModelSelect />
          <ParamVAEPrecision />
        </Flex>
        <FormControlGroup formLabelProps={formLabelProps}>
          <ParamClipSkip />
          <ParamCFGRescaleMultiplier />
        </FormControlGroup>
        <Flex gap={4} w="full">
          <FormControlGroup formLabelProps={formLabelProps2}>
            <ParamSeamlessXAxis />
            <ParamSeamlessYAxis />
          </FormControlGroup>
        </Flex>
      </Flex>
    </StandaloneAccordion>
  );
});

AdvancedSettingsAccordion.displayName = 'AdvancedSettingsAccordion';
