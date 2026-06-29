import { ImageMetadataHandlers } from 'features/metadata/parsing';
import { describe, expect, it } from 'vitest';

import { IMAGE_METADATA_ACTION_HANDLERS } from './ImageMetadataActions';

describe('IMAGE_METADATA_ACTION_HANDLERS', () => {
  it('includes Qwen metadata handlers in the recall parameters UI', () => {
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageComponentSource);
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageQuantization);
    expect(IMAGE_METADATA_ACTION_HANDLERS).toContain(ImageMetadataHandlers.QwenImageShift);
  });
});
