import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import { getNodeSubtitle } from './nodeSummaries';

const t = ((key: string, options?: Record<string, unknown>) => `${key}:${JSON.stringify(options ?? {})}`) as TFunction;

describe('getNodeSubtitle', () => {
  it('returns the denoise summary when steps and cfg_scale are numbers', () => {
    const subtitle = getNodeSubtitle(
      { id: 'denoise_latents', inputs: { cfg_scale: 4, steps: 28 }, type: 'denoise_latents' },
      t
    );

    expect(subtitle).toBe('graphPreview.nodeSummary.denoise:{"cfg":4,"steps":28}');
  });

  it('returns the noise summary when width and height are numbers', () => {
    const subtitle = getNodeSubtitle({ id: 'noise', inputs: { height: 512, width: 768 }, type: 'noise' }, t);

    expect(subtitle).toBe('graphPreview.nodeSummary.noise:{"height":512,"width":768}');
  });

  it('returns null for denoise_latents when the numeric fields are missing', () => {
    const subtitle = getNodeSubtitle({ id: 'denoise_latents', inputs: {}, type: 'denoise_latents' }, t);

    expect(subtitle).toBeNull();
  });

  it('returns null for an unknown node type', () => {
    const subtitle = getNodeSubtitle({ id: 'l2i', inputs: {}, type: 'l2i' }, t);

    expect(subtitle).toBeNull();
  });
});
