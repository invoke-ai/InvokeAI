import { createMemoizedAppSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { layerUsedAsControlChanged, selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { selectLayer } from 'features/controlLayers/store/layersReducers';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { initialControlNetV2, initialT2IAdapterV2 } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useCallback, useMemo } from 'react';
import { useControlNetAndT2IAdapterModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

export const useLayerControlAdapter = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectControlAdapter = useMemo(
    () =>
      createMemoizedAppSelector(selectCanvasV2Slice, (canvasV2) => {
        const layer = selectLayer(canvasV2, entityIdentifier.id);
        if (!layer) {
          return null;
        }
        return layer.controlAdapter;
      }),
    [entityIdentifier]
  );
  const controlAdapter = useAppSelector(selectControlAdapter);
  return controlAdapter;
};

export const useLayerUseAsControl = (entityIdentifier: CanvasEntityIdentifier) => {
  const dispatch = useAppDispatch();
  const [modelConfigs] = useControlNetAndT2IAdapterModels();

  const baseModel = useAppSelector((s) => s.canvasV2.params.model?.base);
  const controlAdapter = useLayerControlAdapter(entityIdentifier);

  const model: ControlNetModelConfig | T2IAdapterModelConfig | null = useMemo(() => {
    // prefer to use a model that matches the base model
    const compatibleModels = modelConfigs.filter((m) => (baseModel ? m.base === baseModel : true));
    return compatibleModels[0] ?? modelConfigs[0] ?? null;
  }, [baseModel, modelConfigs]);

  const toggle = useCallback(() => {
    if (controlAdapter) {
      dispatch(layerUsedAsControlChanged({ id: entityIdentifier.id, controlAdapter: null }));
      return;
    }
    const newControlAdapter = deepClone(model?.type === 't2i_adapter' ? initialT2IAdapterV2 : initialControlNetV2);

    if (model) {
      newControlAdapter.model = zModelIdentifierField.parse(model);
    }

    dispatch(layerUsedAsControlChanged({ id: entityIdentifier.id, controlAdapter: newControlAdapter }));
  }, [controlAdapter, dispatch, entityIdentifier.id, model]);

  return { hasControlAdapter: Boolean(controlAdapter), toggle };
};
