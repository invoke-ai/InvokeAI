import { logger } from 'app/logging/logger';
import { withResultAsync } from 'common/util/result';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasControlLayerState, Rect } from 'features/controlLayers/store/types';
import { getControlLayerWarnings } from 'features/controlLayers/store/validators';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { serializeError } from 'serialize-error';
import type { ImageDTO, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

type AddControlNetsArg = {
  manager: CanvasManager;
  entities: CanvasControlLayerState[];
  g: Graph;
  rect: Rect;
  collector: Invocation<'collect'>;
  model: ParameterModel;
};

type AddControlNetsResult = {
  addedControlNets: number;
};

export const addControlNets = async ({
  manager,
  entities,
  g,
  rect,
  collector,
  model,
}: AddControlNetsArg): Promise<AddControlNetsResult> => {
  const validControlLayers = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => entity.controlAdapter.type === 'controlnet')
    .filter((entity) => getControlLayerWarnings(entity, model).length === 0);

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

type AddT2IAdaptersArg = {
  manager: CanvasManager;
  entities: CanvasControlLayerState[];
  g: Graph;
  rect: Rect;
  collector: Invocation<'collect'>;
  model: ParameterModel;
};

type AddT2IAdaptersResult = {
  addedT2IAdapters: number;
};

export const addT2IAdapters = async ({
  manager,
  entities,
  g,
  rect,
  collector,
  model,
}: AddT2IAdaptersArg): Promise<AddT2IAdaptersResult> => {
  const validControlLayers = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => entity.controlAdapter.type === 't2i_adapter')
    .filter((entity) => getControlLayerWarnings(entity, model).length === 0);

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

type AddControlLoRAArg = {
  manager: CanvasManager;
  entities: CanvasControlLayerState[];
  g: Graph;
  rect: Rect;
  model: ParameterModel;
  denoise: Invocation<'flux_denoise'>;
};

export const addControlLoRA = async ({ manager, entities, g, rect, model, denoise }: AddControlLoRAArg) => {
  const validControlLayers = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => entity.controlAdapter.type === 'control_lora')
    .filter((entity) => getControlLayerWarnings(entity, model).length === 0);

  const validControlLayer = validControlLayers[0];
  if (validControlLayer === undefined) {
    // No valid control LoRA found
    return;
  }
  if (validControlLayers.length > 1) {
    throw new Error('Cannot add more than one FLUX control LoRA.');
  }

  const getImageDTOResult = await withResultAsync(() => {
    const adapter = manager.adapters.controlLayers.get(validControlLayer.id);
    assert(adapter, 'Adapter not found');
    return adapter.renderer.rasterize({ rect, attrs: { opacity: 1, filters: [] }, bg: 'black' });
  });
  if (getImageDTOResult.isErr()) {
    log.warn({ error: serializeError(getImageDTOResult.error) }, 'Error rasterizing control layer');
    return;
  }

  const imageDTO = getImageDTOResult.value;
  addControlLoRAToGraph(g, validControlLayer, imageDTO, denoise);
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

const addControlLoRAToGraph = (
  g: Graph,
  layer: CanvasControlLayerState,
  imageDTO: ImageDTO,
  denoise: Invocation<'flux_denoise'>
) => {
  const { id, controlAdapter } = layer;
  assert(controlAdapter.type === 'control_lora');
  const { model, weight } = controlAdapter;
  assert(model !== null);
  const { image_name } = imageDTO;

  const controlLoRA = g.addNode({
    id: `control_lora_${id}`,
    type: 'flux_control_lora_loader',
    lora: model,
    image: { image_name },
    weight: weight,
  });

  g.addEdge(controlLoRA, 'control_lora', denoise, 'control_lora');
};
