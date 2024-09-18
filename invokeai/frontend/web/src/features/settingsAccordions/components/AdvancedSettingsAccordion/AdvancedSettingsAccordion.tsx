import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsFLUX, selectParamsSlice, selectVAEKey } from 'features/controlLayers/store/paramsSlice';
import ParamCFGRescaleMultiplier from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import ParamCLIPEmbedModelSelect from 'features/parameters/components/Advanced/ParamCLIPEmbedModelSelect';
import ParamClipSkip from 'features/parameters/components/Advanced/ParamClipSkip';
import ParamT5EncoderModelSelect from 'features/parameters/components/Advanced/ParamT5EncoderModelSelect';
import ParamSeamlessXAxis from 'features/parameters/components/Seamless/ParamSeamlessXAxis';
import ParamSeamlessYAxis from 'features/parameters/components/Seamless/ParamSeamlessYAxis';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import ParamFLUXVAEModelSelect from 'features/parameters/components/VAEModel/ParamFLUXVAEModelSelect';
import ParamVAEModelSelect from 'features/parameters/components/VAEModel/ParamVAEModelSelect';
import ParamVAEPrecision from 'features/parameters/components/VAEModel/ParamVAEPrecision';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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
  const activeTabName = useAppSelector(selectActiveTab);
  const isFLUX = useAppSelector(selectIsFLUX);

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
          if (activeTabName === 'upscaling' && !params.shouldRandomizeSeed) {
            badges.push('Manual Seed');
          }
        }

        return badges;
      }),
    [vaeConfig, activeTabName]
  );
  const badges = useAppSelector(selectBadges);
  const { t } = useTranslation();
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: `'advanced-settings-${activeTabName}`,
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion label={t('accordions.advanced.title')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex gap={4} alignItems="center" p={4} flexDir="column" data-testid="advanced-settings-accordion">
        <Flex gap={4} w="full">
          {isFLUX ? <ParamFLUXVAEModelSelect /> : <ParamVAEModelSelect />}
          {!isFLUX && <ParamVAEPrecision />}
        </Flex>
        {activeTabName === 'upscaling' ? (
          <Flex gap={4} alignItems="center">
            <ParamSeedNumberInput />
            <ParamSeedShuffle />
            <ParamSeedRandomize />
          </Flex>
        ) : (
          <>
            {!isFLUX && (
              <>
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
              </>
            )}
            {isFLUX && (
              <FormControlGroup>
                <ParamT5EncoderModelSelect />
                <ParamCLIPEmbedModelSelect />
              </FormControlGroup>
            )}
          </>
        )}
      </Flex>
    </StandaloneAccordion>
  );
});

AdvancedSettingsAccordion.displayName = 'AdvancedSettingsAccordion';
