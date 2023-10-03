import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSearchableSelect';
import { configSelector } from 'features/system/store/configSelectors';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';
import { CONTROLNET_PROCESSORS } from '../../store/constants';
import {
  ControlNetConfig,
  controlNetProcessorTypeChanged,
} from '../../store/controlNetSlice';
import { ControlNetProcessorType } from '../../store/types';
import { useTranslation } from 'react-i18next';

type ParamControlNetProcessorSelectProps = {
  controlNet: ControlNetConfig;
};

const selector = createSelector(
  configSelector,
  (config) => {
    const controlNetProcessors: IAISelectDataType[] = map(
      CONTROLNET_PROCESSORS,
      (p) => ({
        value: p.type,
        label: p.label,
      })
    )
      .sort((a, b) =>
        // sort 'none' to the top
        a.value === 'none'
          ? -1
          : b.value === 'none'
          ? 1
          : a.label.localeCompare(b.label)
      )
      .filter(
        (d) =>
          !config.sd.disabledControlNetProcessors.includes(
            d.value as ControlNetProcessorType
          )
      );

    return controlNetProcessors;
  },
  defaultSelectorOptions
);

const ParamControlNetProcessorSelect = (
  props: ParamControlNetProcessorSelectProps
) => {
  const dispatch = useAppDispatch();
  const { controlNetId, isEnabled, processorNode } = props.controlNet;
  const controlNetProcessors = useAppSelector(selector);
  const { t } = useTranslation();

  const handleProcessorTypeChanged = useCallback(
    (v: string | null) => {
      dispatch(
        controlNetProcessorTypeChanged({
          controlNetId,
          processorType: v as ControlNetProcessorType,
        })
      );
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      label={t('controlnet.processor')}
      value={processorNode.type ?? 'canny_image_processor'}
      data={controlNetProcessors}
      onChange={handleProcessorTypeChanged}
      disabled={!isEnabled}
    />
  );
};

export default memo(ParamControlNetProcessorSelect);
