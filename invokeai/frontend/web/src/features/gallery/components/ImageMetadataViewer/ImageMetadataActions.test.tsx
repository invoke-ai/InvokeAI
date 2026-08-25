import { ImageMetadataHandlers } from 'features/metadata/parsing';
import { describe, expect, it } from 'vitest';

import { IMAGE_METADATA_ACTION_HANDLERS } from './ImageMetadataActions';

describe('IMAGE_METADATA_ACTION_HANDLERS', () => {
  it('includes Qwen metadata handlers in the recall parameters UI', () => {
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageComponentSource);
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageQuantization);
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageShift);
  });

  it('includes every Krea-2 metadata handler in the recall parameters UI', () => {
    // Krea-2 records standalone components (single-file / GGUF) and the conditioning-enhancer settings.
    // All must be wired into the recall UI, otherwise they are saved to metadata but cannot be recalled.
    const krea2Handlers = [
      ImageMetadataHandlers.Krea2VAEModel,
      ImageMetadataHandlers.Krea2Qwen3VlEncoderModel,
      ImageMetadataHandlers.Krea2SeedVarianceEnabled,
      ImageMetadataHandlers.Krea2SeedVarianceStrength,
      ImageMetadataHandlers.Krea2SeedVarianceRandomizePercent,
      ImageMetadataHandlers.Krea2RebalanceEnabled,
      ImageMetadataHandlers.Krea2RebalanceMultiplier,
      ImageMetadataHandlers.Krea2RebalanceWeights,
    ];
    for (const handler of krea2Handlers) {
      expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(handler);
    }
  });

  it('includes every standalone-component handler for FLUX.1, Z-Image and Anima', () => {
    // These bases keep their VAE / text encoder in dedicated params slots and write them to metadata,
    // but the handlers were missing from this list — so the values were saved and gated correctly yet
    // never rendered a row or a per-parameter recall button.
    const handlers = [
      ImageMetadataHandlers.Flux1VAEModel,
      ImageMetadataHandlers.ZImageVAEModel,
      ImageMetadataHandlers.ZImageQwen3EncoderModel,
      ImageMetadataHandlers.ZImageQwen3SourceModel,
      ImageMetadataHandlers.AnimaVAEModel,
      ImageMetadataHandlers.AnimaQwen3EncoderModel,
    ];
    for (const handler of handlers) {
      expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(handler);
    }
  });

  it('has no un-triaged handler missing from the recall parameters UI', () => {
    // Guard against the list drifting behind the handler registry again: every new handler must either be
    // listed above or be added here with a reason. The second group is a snapshot of pre-existing gaps —
    // metadata that is written but has no recall row. Shrink it, don't grow it.
    const intentionallyHidden = new Set<string>([
      // Not per-parameter recallable by design.
      'CreatedBy', // informational only
      'ImageSize', // combined width+height, driven by the dedicated "recall size" action

      // Known gaps, not triaged in this change.
      'HiDiffusion',
      'HiDiffusionRauNet',
      'HiDiffusionWindowAttn',
      'HiDiffusionT1Ratio',
      'HiDiffusionT2Ratio',
      'ZImageSeedVarianceEnabled',
      'ZImageSeedVarianceStrength',
      'ZImageSeedVarianceRandomizePercent',
      'QwenImageVaeModel',
      'QwenImageQwenVLEncoderModel',
      'WanTransformerLowNoise',
      'WanComponentSource',
      'WanVaeModel',
      'WanT5EncoderModel',
      'WanGuidanceScaleLowNoise',
      'GeminiTemperature',
      'GeminiThinkingLevel',
      'OpenaiQuality',
      'OpenaiBackground',
      'OpenaiInputFidelity',
      'SeedreamWatermark',
      'SeedreamOptimizePrompt',
    ]);

    const missing = Object.values(ImageMetadataHandlers)
      .filter((handler) => !IMAGE_METADATA_ACTION_HANDLERS.includes(handler) && !intentionallyHidden.has(handler.type))
      .map((handler) => handler.type);

    expect(missing).toEqual([]);
  });
});
