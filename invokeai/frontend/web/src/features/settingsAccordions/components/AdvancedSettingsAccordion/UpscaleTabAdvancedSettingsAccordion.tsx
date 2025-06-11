import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsFLUX, selectIsSD3, selectParamsSlice, selectVAEKey } from 'features/controlLayers/store/paramsSlice';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import ParamFLUXVAEModelSelect from 'features/parameters/components/VAEModel/ParamFLUXVAEModelSelect';
import ParamVAEModelSelect from 'features/parameters/components/VAEModel/ParamVAEModelSelect';
import ParamVAEPrecision from 'features/parameters/components/VAEModel/ParamVAEPrecision';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

const formLabelProps: FormLabelProps = {
  minW: '9.2rem',
};

const formLabelProps2: FormLabelProps = {
  flexGrow: 1,
};

export const AdvancedSettingsAccordion = memo(() => {
  const vaeKey = useAppSelector(selectVAEKey);
  const { currentData: vaeConfig } = useGetModelConfigQuery(vaeKey ?? skipToken);
  const isFLUX = useAppSelector(selectIsFLUX);
  const isSD3 = useAppSelector(selectIsSD3);

  const selectBadges = useMemo(
    () =>
      createMemoizedSelector([selectParamsSlice, selectIsFLUX], (params, isFLUX) => {
        const badges: (string | number)[] = [];
        if (isFLUX) {
          if (vaeConfig) {
            let vaeBadge = vaeConfig.name;
            if (params.vaePrecision === 'fp16') {
              vaeBadge += ` ${params.vaePrecision}`;
            }
            badges.push(vaeBadge);
          }
        } else {
          if (vaeConfig) {
            let vaeBadge = vaeConfig.name;
            if (params.vaePrecision === 'fp16') {
              vaeBadge += ` ${params.vaePrecision}`;
            }
            badges.push(vaeBadge);
          } else if (params.vaePrecision === 'fp16') {
            badges.push(`VAE ${params.vaePrecision}`);
          }
          if (params.clipSkip) {
            badges.push(`Skip ${params.clipSkip}`);
          }
          if (params.cfgRescaleMultiplier) {
            badges.push(`Rescale ${params.cfgRescaleMultiplier}`);
          }
          if (params.seamlessXAxis || params.seamlessYAxis) {
            badges.push('seamless');
          }
          if (!params.shouldRandomizeSeed) {
            badges.push('Manual Seed');
          }
        }

        return badges;
      }),
    [vaeConfig]
  );
  const badges = useAppSelector(selectBadges);
  const { t } = useTranslation();
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: `'advanced-settings-upscaling`,
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion label={t('accordions.advanced.title')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex gap={4} alignItems="center" p={4} flexDir="column" data-testid="advanced-settings-accordion">
        <Flex gap={4} w="full">
          {isFLUX ? <ParamFLUXVAEModelSelect /> : <ParamVAEModelSelect />}
          {!isFLUX && !isSD3 && <ParamVAEPrecision />}
        </Flex>
        <ParamSeed />
      </Flex>
    </StandaloneAccordion>
  );
});

AdvancedSettingsAccordion.displayName = 'AdvancedSettingsAccordion';
