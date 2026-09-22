import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, MainModelConfig, S } from 'services/api/types';
import { isWanLoRAModelConfig } from 'services/api/types';

const log = logger('system');

/** Map a Wan main-model variant onto the LoRA-variant string used by the
 *  probe. A14B (both T2V and I2V) uses inner_dim=5120 → "a14b". TI2V-5B
 *  uses inner_dim=3072 → "5b". */
const mainVariantToLoRAVariant = (mainVariant: string | null | undefined): 'a14b' | '5b' | null => {
  if (mainVariant === 't2v_a14b' || mainVariant === 'i2v_a14b') {
    return 'a14b';
  }
  if (mainVariant === 'ti2v_5b') {
    return '5b';
  }
  return null;
};

/**
 * Add Wan 2.2 LoRA wiring to the graph between the model loader and the
 * denoise node.
 *
 * Each enabled Wan LoRA becomes a ``lora_selector`` feeding a ``collect``
 * node, which fans into a ``wan_lora_collection_loader``. The collection
 * loader rewrites the model loader's transformer output into a
 * ``WanTransformerField`` with the appropriate ``loras`` /
 * ``loras_low_noise`` lists populated based on each LoRA's recorded
 * ``expert`` tag — high-noise LoRAs land on the primary list, low-noise
 * LoRAs on ``loras_low_noise``, and untagged LoRAs are applied to both
 * experts. The dual-expert routing happens entirely on the backend; the
 * FE just hands the loader the bag of LoRAs.
 *
 * Variant filter: each LoRA's full config carries a ``variant`` field
 * (``"a14b"`` / ``"5b"`` / null) set by the backend probe from the LoRA's
 * inner-dim. A14B LoRAs have 5120-dim weights and can't be reshaped to
 * fit a TI2V-5B main (3072-dim) — the layer patcher would crash with a
 * tensor-size error. We fetch each LoRA's config and skip mismatches,
 * logging a warning so the user can tell why a LoRA they enabled didn't
 * take effect.
 */
export const addWanLoRAs = async (
  state: RootState,
  g: Graph,
  denoise: Invocation<'wan_denoise'>,
  modelLoader: Invocation<'wan_model_loader'>,
  mainConfig: MainModelConfig
): Promise<void> => {
  // MainModelConfig is the union of all main-config schemas; ``variant`` is
  // only present on the discriminated members (Wan, FLUX, ZImage, etc.).
  // Read it defensively rather than relying on TypeScript narrowing through
  // a typed parameter.
  const mainVariant = 'variant' in mainConfig ? ((mainConfig as { variant?: string | null }).variant ?? null) : null;
  const expectedLoRAVariant = mainVariantToLoRAVariant(mainVariant);
  const candidateLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'wan');

  if (candidateLoRAs.length === 0) {
    return;
  }

  // Fetch each LoRA's config and filter by variant compatibility. LoRAs
  // with ``variant === null`` are kept (the probe couldn't identify them;
  // best to try rather than silently drop).
  const compatibleLoRAs: typeof candidateLoRAs = [];
  for (const lora of candidateLoRAs) {
    try {
      const cfg = await fetchModelConfigWithTypeGuard(lora.model.key, isWanLoRAModelConfig);
      const loraVariant = cfg.variant ?? null;
      if (loraVariant !== null && expectedLoRAVariant !== null && loraVariant !== expectedLoRAVariant) {
        log.warn(
          { lora: lora.model.name, loraVariant, mainVariant },
          `Skipping Wan LoRA "${lora.model.name}" — its variant (${loraVariant}) is incompatible with ` +
            `the selected main model variant (${mainVariant}). ` +
            `A14B and TI2V-5B have different inner dims and LoRA weights aren't interchangeable.`
        );
        continue;
      }
      compatibleLoRAs.push(lora);
    } catch (e) {
      // If the config can't be fetched, fall back to including the LoRA —
      // the backend will produce a clearer error if it really doesn't fit.
      log.warn({ lora: lora.model.name, error: String(e) }, `Failed to read variant for Wan LoRA "${lora.model.name}"`);
      compatibleLoRAs.push(lora);
    }
  }

  if (compatibleLoRAs.length === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'wan_lora_collection_loader',
    id: getPrefixedId('wan_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.deleteEdgesTo(denoise, ['transformer']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');

  for (const lora of compatibleLoRAs) {
    const { weight } = lora;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: getPrefixedId('lora_selector'),
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({
      model: parsedModel,
      weight,
    });

    g.addEdge(loraSelector, 'lora', loraCollector, 'item');
  }

  g.upsertMetadata({ loras: loraMetadata });
};
