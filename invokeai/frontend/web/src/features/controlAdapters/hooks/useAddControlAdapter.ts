import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CONTROLNET_MODEL_DEFAULT_PROCESSORS, CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import type {
  ControlAdapterProcessorType,
  ControlAdapterType,
  RequiredControlAdapterProcessorNode,
} from 'features/controlAdapters/store/types';
import { cloneDeep } from 'lodash-es';
import { useCallback, useMemo } from 'react';

import { useControlAdapterModels } from './useControlAdapterModels';

export const useAddControlAdapter = (type: ControlAdapterType) => {
  const baseModel = useAppSelector((s) => s.generation.model?.base);
  const dispatch = useAppDispatch();

  const models = useControlAdapterModels(type);

  const firstModel = useMemo(() => {
    // prefer to use a model that matches the base model
    const firstCompatibleModel = models.filter((m) => (baseModel ? m.base === baseModel : true))[0];

    if (firstCompatibleModel) {
      return firstCompatibleModel;
    }

    return models[0];
  }, [baseModel, models]);

  const processor = useMemo(() => {
    let processorType;
    for (const modelSubstring in CONTROLNET_MODEL_DEFAULT_PROCESSORS) {
      if (firstModel?.name.includes(modelSubstring)) {
        processorType = CONTROLNET_MODEL_DEFAULT_PROCESSORS[modelSubstring];
        break;
      }
    }

    if (!processorType) {
      processorType = 'none';
    }

    const processorNode =
      processorType === 'none'
        ? (cloneDeep(CONTROLNET_PROCESSORS.none.default) as RequiredControlAdapterProcessorNode)
        : (cloneDeep(
            CONTROLNET_PROCESSORS[processorType as ControlAdapterProcessorType].default
          ) as RequiredControlAdapterProcessorNode);

    return { processorType, processorNode };
  }, [firstModel]);

  const isDisabled = useMemo(() => !firstModel, [firstModel]);

  const addControlAdapter = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(
      controlAdapterAdded({
        type,
        overrides: { model: firstModel, ...processor },
      })
    );
  }, [dispatch, firstModel, isDisabled, type, processor]);

  return [addControlAdapter, isDisabled] as const;
};
