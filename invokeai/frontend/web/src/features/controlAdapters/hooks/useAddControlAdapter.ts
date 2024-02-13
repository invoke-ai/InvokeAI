import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { controlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterType } from 'features/controlAdapters/store/types';
import { useCallback, useMemo } from 'react';

import { useControlAdapterModels } from './useControlAdapterModels';

export const useAddControlAdapter = (type: ControlAdapterType) => {
  const baseModel = useAppSelector((s) => s.generation.model?.base_model);
  const dispatch = useAppDispatch();

  const models = useControlAdapterModels(type);

  const firstModel = useMemo(() => {
    // prefer to use a model that matches the base model
    const firstCompatibleModel = models.filter((m) => (baseModel ? m.base_model === baseModel : true))[0];

    if (firstCompatibleModel) {
      return firstCompatibleModel;
    }

    return models[0];
  }, [baseModel, models]);

  const isDisabled = useMemo(() => !firstModel, [firstModel]);

  const addControlAdapter = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(
      controlAdapterAdded({
        type,
        overrides: { model: firstModel },
      })
    );
  }, [dispatch, firstModel, isDisabled, type]);

  return [addControlAdapter, isDisabled] as const;
};
