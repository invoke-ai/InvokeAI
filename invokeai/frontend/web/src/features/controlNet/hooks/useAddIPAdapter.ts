import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  controlAdapterAdded,
  selectAllIPAdapters,
} from 'features/controlNet/store/controlAdaptersSlice';
import { useCallback, useMemo } from 'react';
import {
  ipAdapterModelsAdapter,
  useGetIPAdapterModelsQuery,
} from 'services/api/endpoints/models';

const selector = createSelector(
  [stateSelector],
  ({ controlAdapters, generation }) => {
    const ipAdapterCount = selectAllIPAdapters(controlAdapters).length;
    const { model } = generation;
    return {
      ipAdapterCount,
      baseModel: model?.base_model,
    };
  },
  defaultSelectorOptions
);

export const useAddIPAdapter = () => {
  const { ipAdapterCount, baseModel } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery();
  const firstIPAdapterModel = useMemo(
    () =>
      ipAdapterModels
        ? ipAdapterModelsAdapter
            .getSelectors()
            .selectAll(ipAdapterModels)
            .filter((m) => (baseModel ? m.base_model === baseModel : true))[0]
        : undefined,
    [baseModel, ipAdapterModels]
  );
  const isDisabled = useMemo(
    () => !firstIPAdapterModel && ipAdapterCount === 0,
    [firstIPAdapterModel, ipAdapterCount]
  );
  const addIPAdapter = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(
      controlAdapterAdded({
        type: 'ip_adapter',
        overrides: { model: firstIPAdapterModel },
      })
    );
  }, [dispatch, firstIPAdapterModel, isDisabled]);

  return {
    addIPAdapter,
    isDisabled,
  };
};
