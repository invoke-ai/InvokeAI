import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { controlAdapterAdded } from 'features/controlNet/store/controlAdaptersSlice';
import { useCallback, useMemo } from 'react';
import {
  controlNetModelsAdapter,
  useGetControlNetModelsQuery,
} from 'services/api/endpoints/models';

export const useAddControlNet = () => {
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector(
    (state) => state.generation.model?.base_model
  );
  const { data: controlNetModels } = useGetControlNetModelsQuery();
  const firstControlNetModel = useMemo(
    () =>
      controlNetModels
        ? controlNetModelsAdapter
            .getSelectors()
            .selectAll(controlNetModels)
            .filter((m) => (baseModel ? m.base_model === baseModel : true))[0]
        : undefined,
    [baseModel, controlNetModels]
  );
  const isDisabled = useMemo(
    () => !firstControlNetModel,
    [firstControlNetModel]
  );
  const addControlNet = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(
      controlAdapterAdded({
        type: 'controlnet',
        overrides: { model: firstControlNetModel },
      })
    );
  }, [dispatch, firstControlNetModel, isDisabled]);

  return {
    addControlNet,
    isDisabled,
  };
};
