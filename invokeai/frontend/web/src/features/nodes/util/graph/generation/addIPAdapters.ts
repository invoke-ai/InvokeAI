import { type IPAdapterConfig, isIPAdapterConfig, type RefImageState } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, MainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type AddIPAdaptersResult = {
  addedIPAdapters: number;
};

type AddIPAdaptersArg = {
  entities: RefImageState[];
  g: Graph;
  collector: Invocation<'collect'>;
  model: MainModelConfig;
};

export const addIPAdapters = ({ entities, g, collector, model }: AddIPAdaptersArg): AddIPAdaptersResult => {
  const validIPAdapters = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isIPAdapterConfig(entity.config))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

  const result: AddIPAdaptersResult = {
    addedIPAdapters: 0,
  };

  for (const { id, config } of validIPAdapters) {
    assert(isIPAdapterConfig(config), 'This should have been filtered out');
    result.addedIPAdapters++;

    addIPAdapter(id, config, g, collector);
  }

  g.upsertMetadata({ ref_images: validIPAdapters }, 'merge');

  return result;
};

const addIPAdapter = (id: string, ipAdapter: IPAdapterConfig, g: Graph, collector: Invocation<'collect'>) => {
  const { weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;
  assert(image, 'IP Adapter image is required');
  assert(model, 'IP Adapter model is required');

  let ipAdapterNode: Invocation<'flux_ip_adapter' | 'ip_adapter'>;

  if (model.base === 'flux') {
    assert(
      clipVisionModel === 'ViT-L',
      `ViT-L is the only supported CLIP Vision model for FLUX IP adapter, got ${clipVisionModel}`
    );
    ipAdapterNode = g.addNode({
      id: `ip_adapter_${id}`,
      type: 'flux_ip_adapter',
      weight,
      ip_adapter_model: model,
      clip_vision_model: clipVisionModel,
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      image: {
        image_name: image.image_name,
      },
    });
  } else {
    // model.base === SD1.5 or SDXL
    assert(
      clipVisionModel === 'ViT-H' || clipVisionModel === 'ViT-G',
      'ViT-G and ViT-H are the only supported CLIP Vision models for SD1.5 and SDXL IP adapters'
    );
    ipAdapterNode = g.addNode({
      id: `ip_adapter_${id}`,
      type: 'ip_adapter',
      weight,
      method,
      ip_adapter_model: model,
      clip_vision_model: clipVisionModel,
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      image: {
        image_name: image.image_name,
      },
    });
  }

  g.addEdge(ipAdapterNode, 'ip_adapter', collector, 'item');
};
