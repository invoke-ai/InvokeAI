import { logger } from 'app/logging/logger';
import { withResultAsync } from 'common/util/result';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type {
  CanvasControlLayerState,
  ControlNetConfig,
  Rect,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { serializeError } from 'serialize-error';
import type { BaseModelType, ImageDTO, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

type AddControlNetsResult = {
  addedControlNets: number;
};

export const addControlNets = async (
  manager: CanvasManager,
  layers: CanvasControlLayerState[],
  g: Graph,
  rect: Rect,
  collector: Invocation<'collect'>,
  base: BaseModelType
): Promise<AddControlNetsResult> => {
  const validControlLayers = layers
    .filter((layer) => layer.isEnabled)
    .filter((layer) => isValidControlAdapter(layer.controlAdapter, base))
    .filter((layer) => layer.controlAdapter.type === 'controlnet');

  const result: AddControlNetsResult = {
    addedControlNets: 0,
  };

  for (const layer of validControlLayers) {
    const getImageDTOResult = await withResultAsync(() => {
      const adapter = manager.adapters.controlLayers.get(layer.id);
      assert(adapter, 'Adapter not found');
      return adapter.renderer.rasterize({ rect, attrs: { opacity: 1, filters: [] }, bg: 'black' });
    });
    if (getImageDTOResult.isErr()) {
      log.warn({ error: serializeError(getImageDTOResult.error) }, 'Error rasterizing control layer');
      continue;
    }

    const imageDTO = getImageDTOResult.value;
    addControlNetToGraph(g, layer, imageDTO, collector);
    result.addedControlNets++;
  }

  return result;
};

type AddT2IAdaptersResult = {
  addedT2IAdapters: number;
};

export const addT2IAdapters = async (
  manager: CanvasManager,
  layers: CanvasControlLayerState[],
  g: Graph,
  rect: Rect,
  collector: Invocation<'collect'>,
  base: BaseModelType
): Promise<AddT2IAdaptersResult> => {
  const validControlLayers = layers
    .filter((layer) => layer.isEnabled)
    .filter((layer) => isValidControlAdapter(layer.controlAdapter, base))
    .filter((layer) => layer.controlAdapter.type === 't2i_adapter');

  const result: AddT2IAdaptersResult = {
    addedT2IAdapters: 0,
  };

  for (const layer of validControlLayers) {
    const getImageDTOResult = await withResultAsync(() => {
      const adapter = manager.adapters.controlLayers.get(layer.id);
      assert(adapter, 'Adapter not found');
      return adapter.renderer.rasterize({ rect, attrs: { opacity: 1, filters: [] }, bg: 'black' });
    });
    if (getImageDTOResult.isErr()) {
      log.warn({ error: serializeError(getImageDTOResult.error) }, 'Error rasterizing control layer');
      continue;
    }

    const imageDTO = getImageDTOResult.value;
    addT2IAdapterToGraph(g, layer, imageDTO, collector);
    result.addedT2IAdapters++;
  }

  return result;
};

const addControlNetToGraph = (
  g: Graph,
  layer: CanvasControlLayerState,
  imageDTO: ImageDTO,
  collector: Invocation<'collect'>
) => {
  const { id, controlAdapter } = layer;
  assert(controlAdapter.type === 'controlnet');
  const { beginEndStepPct, model, weight, controlMode } = controlAdapter;
  assert(model !== null);
  const { image_name } = imageDTO;

  const controlNet = g.addNode({
    id: `control_net_${id}`,
    type: model.base === 'flux' ? 'flux_controlnet' : 'controlnet',
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    control_mode: model.base === 'flux' ? undefined : controlMode,
    resize_mode: 'just_resize',
    control_model: model,
    control_weight: weight,
    image: { image_name },
  });
  g.addEdge(controlNet, 'control', collector, 'item');
};

const addT2IAdapterToGraph = (
  g: Graph,
  layer: CanvasControlLayerState,
  imageDTO: ImageDTO,
  collector: Invocation<'collect'>
) => {
  const { id, controlAdapter } = layer;
  assert(controlAdapter.type === 't2i_adapter');
  const { beginEndStepPct, model, weight } = controlAdapter;
  assert(model !== null);
  const { image_name } = imageDTO;

  const t2iAdapter = g.addNode({
    id: `t2i_adapter_${id}`,
    type: 't2i_adapter',
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    t2i_adapter_model: model,
    weight: weight,
    image: { image_name },
  });

  g.addEdge(t2iAdapter, 't2i_adapter', collector, 'item');
};

const isValidControlAdapter = (controlAdapter: ControlNetConfig | T2IAdapterConfig, base: BaseModelType): boolean => {
  // Must be have a model
  const hasModel = Boolean(controlAdapter.model);
  // Model must match the current base model
  const modelMatchesBase = controlAdapter.model?.base === base;
  return hasModel && modelMatchesBase;
};
