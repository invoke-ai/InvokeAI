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
});
