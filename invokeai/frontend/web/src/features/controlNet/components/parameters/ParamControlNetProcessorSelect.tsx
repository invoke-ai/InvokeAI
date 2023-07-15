import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSearchableSelect';
import { configSelector } from 'features/system/store/configSelectors';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { CONTROLNET_PROCESSORS } from '../../store/constants';
import { controlNetProcessorTypeChanged } from '../../store/controlNetSlice';
import { ControlNetProcessorType } from '../../store/types';

type ParamControlNetProcessorSelectProps = {
  controlNetId: string;
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
  const { controlNetId } = props;
  const processorNodeSelector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { isEnabled, processorNode } =
            controlNet.controlNets[controlNetId];
          return { isEnabled, processorNode };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );
  const isBusy = useAppSelector(selectIsBusy);
  const controlNetProcessors = useAppSelector(selector);
  const { isEnabled, processorNode } = useAppSelector(processorNodeSelector);

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
      label="Processor"
      value={processorNode.type ?? 'canny_image_processor'}
      data={controlNetProcessors}
      onChange={handleProcessorTypeChanged}
      disabled={isBusy || !isEnabled}
    />
  );
};

export default memo(ParamControlNetProcessorSelect);
