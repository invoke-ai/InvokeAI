import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { controlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useCallback, useMemo } from 'react';
import {
  t2iAdapterModelsAdapter,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';

export const useAddT2IAdapter = () => {
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector(
    (state) => state.generation.model?.base_model
  );
  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery();
  const firstT2IAdapterModel = useMemo(
    () =>
      t2iAdapterModels
        ? t2iAdapterModelsAdapter
            .getSelectors()
            .selectAll(t2iAdapterModels)
            .filter((m) => (baseModel ? m.base_model === baseModel : true))[0]
        : undefined,
    [baseModel, t2iAdapterModels]
  );
  const isDisabled = useMemo(
    () => !firstT2IAdapterModel,
    [firstT2IAdapterModel]
  );
  const addT2IAdapter = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(
      controlAdapterAdded({
        type: 't2i_adapter',
        overrides: { model: firstT2IAdapterModel },
      })
    );
  }, [dispatch, firstT2IAdapterModel, isDisabled]);

  return {
    addT2IAdapter,
    isDisabled,
  };
};
