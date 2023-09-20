import { Divider, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamCpuNoiseToggle } from '../Noise/ParamCpuNoise';
import ParamSeamless from '../Seamless/ParamSeamless';
import ParamClipSkip from './ParamClipSkip';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const { clipSkip, seamlessXAxis, seamlessYAxis, shouldUseCpuNoise } =
      state.generation;

    return { clipSkip, seamlessXAxis, seamlessYAxis, shouldUseCpuNoise };
  },
  defaultSelectorOptions
);

export default function ParamAdvancedCollapse() {
  const { clipSkip, seamlessXAxis, seamlessYAxis, shouldUseCpuNoise } =
    useAppSelector(selector);
  const { t } = useTranslation();
  const activeLabel = useMemo(() => {
    const activeLabel: string[] = [];

    if (shouldUseCpuNoise) {
      activeLabel.push(t('parameters.cpuNoise'));
    } else {
      activeLabel.push(t('parameters.gpuNoise'));
    }

    if (clipSkip > 0) {
      activeLabel.push(
        t('parameters.clipSkipWithLayerCount', { layerCount: clipSkip })
      );
    }

    if (seamlessXAxis && seamlessYAxis) {
      activeLabel.push(t('parameters.seamlessX&Y'));
    } else if (seamlessXAxis) {
      activeLabel.push(t('parameters.seamlessX'));
    } else if (seamlessYAxis) {
      activeLabel.push(t('parameters.seamlessY'));
    }

    return activeLabel.join(', ');
  }, [clipSkip, seamlessXAxis, seamlessYAxis, shouldUseCpuNoise, t]);

  return (
    <IAICollapse label={t('common.advanced')} activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamSeamless />
        <Divider />
        <ParamClipSkip />
        <Divider pt={2} />
        <ParamCpuNoiseToggle />
      </Flex>
    </IAICollapse>
  );
}
