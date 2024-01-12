import { Flex } from '@chakra-ui/layout';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import ParamCFGRescaleMultiplier from 'features/parameters/components/Advanced/ParamCFGRescaleMultiplier';
import ParamClipSkip from 'features/parameters/components/Advanced/ParamClipSkip';
import ParamSeamlessXAxis from 'features/parameters/components/Seamless/ParamSeamlessXAxis';
import ParamSeamlessYAxis from 'features/parameters/components/Seamless/ParamSeamlessYAxis';
import ParamVAEModelSelect from 'features/parameters/components/VAEModel/ParamVAEModelSelect';
import ParamVAEPrecision from 'features/parameters/components/VAEModel/ParamVAEPrecision';
import { advancedPanelExpanded } from 'features/parameters/store/actions';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const labelProps: InvLabelProps = {
  minW: '9.2rem',
};

const labelProps2: InvLabelProps = {
  flexGrow: 1,
};

const selectBadges = createMemoizedSelector(
  selectGenerationSlice,
  (generation) => {
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
  }
);

export const AdvancedSettingsAccordion = memo(() => {
  const badges = useAppSelector(selectBadges);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onAccordionClick = useCallback(
    (isOpen?: boolean) => {
      if (!isOpen) {
        dispatch(advancedPanelExpanded());
      }
    },
    [dispatch]
  );

  return (
    <InvSingleAccordion
      label={t('accordions.advanced.title')}
      badges={badges}
      onClick={onAccordionClick}
    >
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
