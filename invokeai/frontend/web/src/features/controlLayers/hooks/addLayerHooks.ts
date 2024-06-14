import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  controlAdapterAdded,
  iiLayerAdded,
  ipAdapterAdded,
  regionalGuidanceIPAdapterAdded,
} from 'features/controlLayers/store/controlLayersSlice';
import { isInitialImageLayer } from 'features/controlLayers/store/types';
import {
  buildControlNet,
  buildIPAdapter,
  buildT2IAdapter,
  CA_PROCESSOR_DATA,
  isProcessorTypeV2,
} from 'features/controlLayers/util/controlAdapters';
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

    dispatch(controlAdapterAdded(controlAdapter));
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
    dispatch(ipAdapterAdded(ipAdapter));
  }, [dispatch, model]);

  return [addIPALayer, isDisabled] as const;
};

export const useAddIPAdapterToIPALayer = (layerId: string) => {
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
    const id = uuidv4();
    const ipAdapter = buildIPAdapter(id, {
      model: zModelIdentifierField.parse(model),
    });
    dispatch(regionalGuidanceIPAdapterAdded({ layerId, ipAdapter }));
  }, [dispatch, model, layerId]);

  return [addIPAdapter, isDisabled] as const;
};

export const useAddIILayer = () => {
  const dispatch = useAppDispatch();
  const isDisabled = useAppSelector((s) => Boolean(s.canvasV2.layers.find(isInitialImageLayer)));
  const addIILayer = useCallback(() => {
    dispatch(iiLayerAdded(null));
  }, [dispatch]);

  return [addIILayer, isDisabled] as const;
};
