import type { TFunction } from 'i18next';

import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import { formatWorkflowPreviewValue, getNodeSubtitle, getWorkflowNodeSummary } from './nodeSummaries';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const englishCatalog = Object.values(englishCatalogModules)[0] as Record<string, unknown>;

/** The English catalog, read as `lng`: numbers follow that language's digits and grouping. */
const translatorFor = async (lng: string): Promise<TFunction> => {
  const i18n = createInstance();
  await i18n.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng,
    resources: { [lng]: { translation: englishCatalog } },
  });

  return i18n.t.bind(i18n) as TFunction;
};

const t = await translatorFor('en');

describe('getNodeSubtitle', () => {
  it('returns the denoise summary when steps and cfg_scale are numbers', () => {
    const subtitle = getNodeSubtitle(
      { id: 'denoise_latents', inputs: { cfg_scale: 4, steps: 28 }, type: 'denoise_latents' },
      t
    );

    expect(subtitle).toBe('28 steps · cfg 4');
  });

  it('reads a single step as one', () => {
    expect(
      getNodeSubtitle({ id: 'denoise_latents', inputs: { cfg_scale: 1, steps: 1 }, type: 'denoise_latents' }, t)
    ).toBe('1 step · cfg 1');
  });

  it('returns the noise summary when width and height are numbers', () => {
    const subtitle = getNodeSubtitle({ id: 'noise', inputs: { height: 512, width: 768 }, type: 'noise' }, t);

    expect(subtitle).toBe('768 × 512');
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

describe('getWorkflowNodeSummary', () => {
  it('reads frames and steps before CFG and steps, and steps alone last', () => {
    expect(getWorkflowNodeSummary({ cfg_scale: 3, num_frames: 121, steps: 30 }, t)).toBe('121 frames · 30 steps');
    expect(getWorkflowNodeSummary({ cfg_scale: 3.5, steps: 30 }, t)).toBe('30 steps · cfg 3.5');
    expect(getWorkflowNodeSummary({ steps: 8 }, t)).toBe('8 steps');
  });

  it('pluralizes frames and steps separately', () => {
    expect(getWorkflowNodeSummary({ frame_count: 1, steps: 1 }, t)).toBe('1 frame · 1 step');
    expect(getWorkflowNodeSummary({ frames: 1, steps: 20 }, t)).toBe('1 frame · 20 steps');
    expect(getWorkflowNodeSummary({ num_frames: 49, steps: 1 }, t)).toBe('49 frames · 1 step');
    expect(getWorkflowNodeSummary({ cfg_scale: 2, steps: 1 }, t)).toBe('1 step · cfg 2');
    expect(getWorkflowNodeSummary({ steps: 1 }, t)).toBe('1 step');
  });

  it('falls back to a size, then to the model the node names, then to nothing', () => {
    expect(getWorkflowNodeSummary({ height: 704, width: 1280 }, t)).toBe('1280 × 704');
    expect(getWorkflowNodeSummary({ model: { key: 'k', name: 'LTX-2 dev' } }, t)).toBe('LTX-2 dev');
    expect(getWorkflowNodeSummary({ prompt: 'a cat', width: 512 }, t)).toBeNull();
  });

  it('writes counts and CFG in the translation language', async () => {
    const german = await translatorFor('de');

    expect(getWorkflowNodeSummary({ num_frames: 1200, steps: 30 }, german)).toBe('1.200 frames · 30 steps');
    expect(getWorkflowNodeSummary({ cfg_scale: 3.5, steps: 30 }, german)).toBe('30 steps · cfg 3,5');
  });
});

describe('formatWorkflowPreviewValue', () => {
  it('writes a number in the translation language, not the browser’s', async () => {
    const value = { kind: 'number', value: 1234.5 } as const;

    expect(formatWorkflowPreviewValue(value, t)).toBe('1,234.5');
    expect(formatWorkflowPreviewValue(value, await translatorFor('de'))).toBe('1.234,5');
  });
});
