import { ImageMetadataHandlers } from 'features/metadata/parsing';
import { describe, expect, it } from 'vitest';

import { ImageMetadataActions } from './ImageMetadataActions';

describe('ImageMetadataActions', () => {
  it('includes Qwen metadata handlers in the recall parameters UI', () => {
    const element = (ImageMetadataActions as unknown as { type: (props: { metadata: unknown }) => unknown }).type({
      metadata: { model: { key: 'test' } },
    }) as {
      props: {
        children: Array<{ props?: { handler?: unknown } }>;
      };
    };

    const handlers = element.props.children
      .map((child) => child.props?.handler)
      .filter((handler): handler is unknown => handler !== undefined);

    expect(handlers).toContain(ImageMetadataHandlers.QwenImageComponentSource);
    expect(handlers).toContain(ImageMetadataHandlers.QwenImageQuantization);
    expect(handlers).toContain(ImageMetadataHandlers.QwenImageShift);
  });
});
