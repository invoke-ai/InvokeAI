import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import { type ControlAdapterType, isControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import { useCallback, useMemo } from 'react';
import type { ControlNetModelConfig, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';

import { useControlAdapterModels } from './useControlAdapterModels';

export const useAddControlAdapter = (type: ControlAdapterType) => {
  const baseModel = useAppSelector((s) => s.generation.model?.base);
  const dispatch = useAppDispatch();

  const models = useControlAdapterModels(type);

  const firstModel: ControlNetModelConfig | T2IAdapterModelConfig | IPAdapterModelConfig | undefined = useMemo(() => {
    // prefer to use a model that matches the base model
    const firstCompatibleModel = models.filter((m) => (baseModel ? m.base === baseModel : true))[0];

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

    if (
      (type === 'controlnet' || type === 't2i_adapter') &&
      (firstModel?.type === 'controlnet' || firstModel?.type === 't2i_adapter')
    ) {
      const defaultPreprocessor = firstModel.default_settings?.preprocessor;
      const processorType = isControlAdapterProcessorType(defaultPreprocessor) ? defaultPreprocessor : 'none';
      const processorNode = CONTROLNET_PROCESSORS[processorType].default;
      dispatch(
        controlAdapterAdded({
          type,
          overrides: {
            model: firstModel,
            processorType,
            processorNode,
          },
        })
      );
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
