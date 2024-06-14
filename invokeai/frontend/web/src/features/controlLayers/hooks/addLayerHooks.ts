import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { caAdded } from 'features/controlLayers/store/controlAdaptersSlice';
import { ipaAdded } from 'features/controlLayers/store/ipAdaptersSlice';
import { rgIPAdapterAdded } from 'features/controlLayers/store/regionalGuidanceSlice';
import {
  buildControlNet,
  buildIPAdapter,
  buildT2IAdapter,
  CA_PROCESSOR_DATA,
  isProcessorTypeV2,
} from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useCallback, useMemo } from 'react';
import { useControlNetAndT2IAdapterModels, useIPAdapterModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

export const useAddCALayer = () => {
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector((s) => s.generation.model?.base);
  const [modelConfigs] = useControlNetAndT2IAdapterModels();
  const model: ControlNetModelConfig | T2IAdapterModelConfig | null = useMemo(() => {
    // prefer to use a model that matches the base model
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    return compatibleModels[0] ?? modelConfigs[0] ?? null;
  }, [baseModel, modelConfigs]);
  const isDisabled = useMemo(() => !model, [model]);
  const addCALayer = useCallback(() => {
    if (!model) {
      return;
    }

    const id = uuidv4();
    const defaultPreprocessor = model.default_settings?.preprocessor;
    const processorConfig = isProcessorTypeV2(defaultPreprocessor)
      ? CA_PROCESSOR_DATA[defaultPreprocessor].buildDefaults(baseModel)
      : null;

    const builder = model.type === 'controlnet' ? buildControlNet : buildT2IAdapter;
    const controlAdapter = builder(id, {
      model: zModelIdentifierField.parse(model),
      processorConfig,
    });

    dispatch(caAdded(controlAdapter));
  }, [dispatch, model, baseModel]);

  return [addCALayer, isDisabled] as const;
};

export const useAddIPALayer = () => {
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector((s) => s.generation.model?.base);
  const [modelConfigs] = useIPAdapterModels();
  const model: IPAdapterModelConfig | null = useMemo(() => {
    // prefer to use a model that matches the base model
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    return compatibleModels[0] ?? modelConfigs[0] ?? null;
  }, [baseModel, modelConfigs]);
  const isDisabled = useMemo(() => !model, [model]);
  const addIPALayer = useCallback(() => {
    if (!model) {
      return;
    }
    const id = uuidv4();
    const ipAdapter = buildIPAdapter(id, {
      model: zModelIdentifierField.parse(model),
    });
    dispatch(ipaAdded(ipAdapter));
  }, [dispatch, model]);

  return [addIPALayer, isDisabled] as const;
};

export const useAddIPAdapterToRGLayer = (id: string) => {
  const dispatch = useAppDispatch();
  const baseModel = useAppSelector((s) => s.generation.model?.base);
  const [modelConfigs] = useIPAdapterModels();
  const model: IPAdapterModelConfig | null = useMemo(() => {
    // prefer to use a model that matches the base model
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    return compatibleModels[0] ?? modelConfigs[0] ?? null;
  }, [baseModel, modelConfigs]);
  const isDisabled = useMemo(() => !model, [model]);
  const addIPAdapter = useCallback(() => {
    if (!model) {
      return;
    }
    const ipAdapter = buildIPAdapter(uuidv4(), {
      model: zModelIdentifierField.parse(model),
    });
    dispatch(rgIPAdapterAdded({ id, ipAdapter: { ...ipAdapter, id: uuidv4(), type: 'ip_adapter', isEnabled: true } }));
  }, [model, dispatch, id]);

  return [addIPAdapter, isDisabled] as const;
};
