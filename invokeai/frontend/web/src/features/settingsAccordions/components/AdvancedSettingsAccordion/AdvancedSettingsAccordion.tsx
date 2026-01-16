import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Box, Flex, FormControlGroup, SimpleGrid, StandaloneAccordion } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectIsFLUX,
  selectIsFlux2,
  selectIsSD3,
  selectIsZImage,
  selectParamsSlice,
  selectVAEKey,
} from 'features/controlLayers/store/paramsSlice';
import ParamCFGRescaleMultiplier from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import ParamCLIPEmbedModelSelect from 'features/parameters/components/Advanced/ParamCLIPEmbedModelSelect';
import ParamCLIPGEmbedModelSelect from 'features/parameters/components/Advanced/ParamCLIPGEmbedModelSelect';
import ParamCLIPLEmbedModelSelect from 'features/parameters/components/Advanced/ParamCLIPLEmbedModelSelect';
import ParamClipSkip from 'features/parameters/components/Advanced/ParamClipSkip';
import ParamT5EncoderModelSelect from 'features/parameters/components/Advanced/ParamT5EncoderModelSelect';
import ParamZImageQwen3VaeModelSelect from 'features/parameters/components/Advanced/ParamZImageQwen3VaeModelSelect';
import ParamSeamlessXAxis from 'features/parameters/components/Seamless/ParamSeamlessXAxis';
import ParamSeamlessYAxis from 'features/parameters/components/Seamless/ParamSeamlessYAxis';
import ParamColorCompensation from 'features/parameters/components/VAEModel/ParamColorCompensation';
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
  const isFlux2 = useAppSelector(selectIsFlux2);
  const isSD3 = useAppSelector(selectIsSD3);
  const isZImage = useAppSelector(selectIsZImage);

  const selectBadges = useMemo(
    () =>
      createMemoizedSelector([selectParamsSlice, selectIsFLUX, selectIsFlux2], (params, isFLUX, isFlux2) => {
        const badges: (string | number)[] = [];
        // FLUX.2 has VAE built into main model - no badge needed
        if (isFLUX && !isFlux2) {
          if (vaeConfig) {
            let vaeBadge = vaeConfig.name;
            if (params.vaePrecision === 'fp16') {
              vaeBadge += ` ${params.vaePrecision}`;
            }
            badges.push(vaeBadge);
          }
        } else if (!isFlux2) {
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
        }

        return badges;
      }),
    [vaeConfig]
  );
  const badges = useAppSelector(selectBadges);
  const { t } = useTranslation();
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: `'advanced-settings-generate`,
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion label={t('accordions.advanced.title')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex gap={4} alignItems="center" p={4} flexDir="column" data-testid="advanced-settings-accordion">
        {!isZImage && !isFlux2 && (
          <Flex gap={4} w="full">
            {isFLUX ? <ParamFLUXVAEModelSelect /> : <ParamVAEModelSelect />}
            {!isFLUX && !isSD3 && <ParamVAEPrecision />}
          </Flex>
        )}
        {!isFLUX && !isFlux2 && !isSD3 && !isZImage && (
          <>
            <FormControlGroup formLabelProps={formLabelProps}>
              <ParamClipSkip />
              <ParamCFGRescaleMultiplier />
            </FormControlGroup>
            <Flex gap={4} w="full">
              <FormControlGroup formLabelProps={formLabelProps2}>
                <SimpleGrid columns={2} spacing={4} w="full">
                  <ParamSeamlessXAxis />
                  <ParamSeamlessYAxis />
                  <ParamColorCompensation />
                  {/* Empty box for visual alignment. Replace with new option when needed. */}
                  <Box />
                </SimpleGrid>
              </FormControlGroup>
            </Flex>
          </>
        )}
        {isFLUX && !isFlux2 && (
          <FormControlGroup>
            <ParamT5EncoderModelSelect />
            <ParamCLIPEmbedModelSelect />
          </FormControlGroup>
        )}
        {/* FLUX.2 Klein: VAE and Qwen3 encoder are extracted from the main model - no selectors needed */}
        {isSD3 && (
          <FormControlGroup>
            <ParamT5EncoderModelSelect />
            <ParamCLIPLEmbedModelSelect />
            <ParamCLIPGEmbedModelSelect />
          </FormControlGroup>
        )}
        {isZImage && (
          <FormControlGroup>
            <ParamZImageQwen3VaeModelSelect />
          </FormControlGroup>
        )}
      </Flex>
    </StandaloneAccordion>
  );
});

AdvancedSettingsAccordion.displayName = 'AdvancedSettingsAccordion';
