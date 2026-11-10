import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import { getNodeSubtitle, getWorkflowNodeSummary } from './nodeSummaries';

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

describe('getWorkflowNodeSummary', () => {
  it('reads frames and steps before CFG and steps, and steps alone last', () => {
    expect(getWorkflowNodeSummary({ cfg_scale: 3, num_frames: 121, steps: 30 }, t)).toBe(
      'workflowLibrary.preview.summaryFramesSteps:{"frames":121,"steps":30}'
    );
    expect(getWorkflowNodeSummary({ cfg_scale: 3, steps: 30 }, t)).toBe(
      'graphPreview.nodeSummary.denoise:{"cfg":3,"steps":30}'
    );
    expect(getWorkflowNodeSummary({ steps: 8 }, t)).toBe('workflowLibrary.preview.summarySteps:{"count":8}');
  });

  it('falls back to a size, then to the model the node names, then to nothing', () => {
    expect(getWorkflowNodeSummary({ height: 704, width: 1280 }, t)).toBe(
      'graphPreview.nodeSummary.noise:{"height":704,"width":1280}'
    );
    expect(getWorkflowNodeSummary({ model: { key: 'k', name: 'LTX-2 dev' } }, t)).toBe('LTX-2 dev');
    expect(getWorkflowNodeSummary({ prompt: 'a cat', width: 512 }, t)).toBeNull();
  });
});
