import { createMemoizedAppSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { selectCanvasV2Slice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  ControlNetConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import { initialControlNet, initialIPAdapter, initialT2IAdapter } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useMemo } from 'react';
import { useControlNetAndT2IAdapterModels, useIPAdapterModels } from 'services/api/hooks/modelsByType';

export const useControlLayerControlAdapter = (entityIdentifier: CanvasEntityIdentifier<'control_layer'>) => {
  const selectControlAdapter = useMemo(
    () =>
      createMemoizedAppSelector(selectCanvasV2Slice, (canvasV2) => {
        const layer = selectEntityOrThrow(canvasV2, entityIdentifier);
        return layer.controlAdapter;
      }),
    [entityIdentifier]
  );
  const controlAdapter = useAppSelector(selectControlAdapter);
  return controlAdapter;
};

/** @knipignore */
export const useDefaultControlAdapter = (): ControlNetConfig | T2IAdapterConfig => {
  const [modelConfigs] = useControlNetAndT2IAdapterModels();

  const baseModel = useAppSelector((s) => s.canvasV2.params.model?.base);

  const defaultControlAdapter = useMemo(() => {
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    const model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    const controlAdapter = model?.type === 't2i_adapter' ? deepClone(initialT2IAdapter) : deepClone(initialControlNet);

    if (model) {
      controlAdapter.model = zModelIdentifierField.parse(model);
    }

    return controlAdapter;
  }, [baseModel, modelConfigs]);

  return defaultControlAdapter;
};

/** @knipignore */
export const useDefaultIPAdapter = (): IPAdapterConfig => {
  const [modelConfigs] = useIPAdapterModels();

  const baseModel = useAppSelector((s) => s.canvasV2.params.model?.base);

  const defaultControlAdapter = useMemo(() => {
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    const model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    const ipAdapter = deepClone(initialIPAdapter);

    if (model) {
      ipAdapter.model = zModelIdentifierField.parse(model);
    }

    return ipAdapter;
  }, [baseModel, modelConfigs]);

  return defaultControlAdapter;
};
