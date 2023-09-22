import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { vaePrecisionChanged } from 'features/parameters/store/generationSlice';
import { PrecisionParam } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback } from 'react';

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { vaePrecision } = generation;
    return { vaePrecision };
  },
  defaultSelectorOptions
);

const DATA = ['fp16', 'fp32'];

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { vaePrecision } = useAppSelector(selector);

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      dispatch(vaePrecisionChanged(v as PrecisionParam));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover details="paramVAEPrecision">
      <IAIMantineSelect
        label="VAE Precision"
        value={vaePrecision}
        data={DATA}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamVAEModelSelect);
