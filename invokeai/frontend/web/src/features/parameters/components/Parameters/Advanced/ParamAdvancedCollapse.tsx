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
import ParamClip2Skip from './ParamClip2Skip';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const {
      clipSkip,
      clip2Skip,
      seamlessXAxis,
      seamlessYAxis,
      shouldUseCpuNoise,
      model,
    } = state.generation;

    return {
      clipSkip,
      clip2Skip,
      seamlessXAxis,
      seamlessYAxis,
      shouldUseCpuNoise,
      shouldShowClip2Skip: model?.base_model === 'sdxl',
    };
  },
  defaultSelectorOptions
);

export default function ParamAdvancedCollapse() {
  const {
    clipSkip,
    clip2Skip,
    seamlessXAxis,
    seamlessYAxis,
    shouldUseCpuNoise,
    shouldShowClip2Skip,
  } = useAppSelector(selector);
  const { t } = useTranslation();
  const activeLabel = useMemo(() => {
    const activeLabel: string[] = [];

    if (shouldUseCpuNoise) {
      activeLabel.push(t('parameters.cpuNoise'));
    } else {
      activeLabel.push(t('parameters.gpuNoise'));
    }

    if ((clipSkip > 0 || clip2Skip > 0) && shouldShowClip2Skip) {
      activeLabel.push(
        t('parameters.clip12SkipWithLayerCount', {
          clipLayerCount: clipSkip,
          clip2LayerCount: clip2Skip,
        })
      );
    } else if (clipSkip > 0) {
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
  }, [
    clip2Skip,
    clipSkip,
    seamlessXAxis,
    seamlessYAxis,
    shouldShowClip2Skip,
    shouldUseCpuNoise,
    t,
  ]);

  return (
    <IAICollapse label={t('common.advanced')} activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamSeamless />
        <Divider />
        <ParamClipSkip />
        <Divider pt={2} />
        {shouldShowClip2Skip && (
          <>
            <ParamClip2Skip />
            <Divider pt={2} />
          </>
        )}

        <ParamCpuNoiseToggle />
      </Flex>
    </IAICollapse>
  );
}
