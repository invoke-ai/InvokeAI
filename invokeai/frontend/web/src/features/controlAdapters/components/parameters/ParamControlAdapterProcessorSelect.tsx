import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSearchableSelect';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterProcessorNode } from 'features/controlAdapters/hooks/useControlAdapterProcessorNode';
import { configSelector } from 'features/system/store/configSelectors';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterProcessortTypeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { ControlAdapterProcessorType } from 'features/controlAdapters/store/types';

type Props = {
  id: string;
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
            d.value as ControlAdapterProcessorType
          )
      );

    return controlNetProcessors;
  },
  defaultSelectorOptions
);

const ParamControlAdapterProcessorSelect = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const processorNode = useControlAdapterProcessorNode(id);
  const dispatch = useAppDispatch();
  const controlNetProcessors = useAppSelector(selector);
  const { t } = useTranslation();

  const handleProcessorTypeChanged = useCallback(
    (v: string | null) => {
      dispatch(
        controlAdapterProcessortTypeChanged({
          id,
          processorType: v as ControlAdapterProcessorType,
        })
      );
    },
    [id, dispatch]
  );

  if (!processorNode) {
    return null;
  }

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

export default memo(ParamControlAdapterProcessorSelect);
