import type { ModelConfig } from '@features/models';

import { buildQueueItemOrigin } from '@features/queue/contracts';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { describe, expect, it, vi } from 'vitest';

import { getLocalQueueItemSource } from './useLocalRecallSnapshot';

vi.mock('@features/queue/devices', async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>;

  return {
    ...original,
    getGenerationDevicesSnapshot: () => ({ error: null, loadState: 'loaded', options: [], setting: 'auto' }),
  };
});

const wanModel: ModelConfig = {
  base: 'wan',
  file_size: 1,
  format: 'diffusers',
  hash: 'wan-hash',
  key: 'wan-t2v_a14b-diffusers',
  name: 'Wan 2.2 t2v_a14b',
  path: 'wan-t2v_a14b-diffusers',
  source: 'wan-t2v_a14b-diffusers',
  source_type: 'path',
  type: 'main',
  variant: 't2v_a14b',
} as ModelConfig;

/** A real state with one submitted queue item, produced by the reducer rather than hand-rolled. */
const submitVideo = () => {
  let state = createInitialWorkbenchState();

  // Video is no longer placed by the non-video defaults; add it first.
  state = workbenchReducer(state, { region: 'left', type: 'toggleRegionWidget', widgetId: 'video' });
  state = workbenchReducer(state, {
    type: 'patchWidgetValues',
    values: { model: wanModel, positivePrompt: 'a fox running' },
    widgetId: 'video',
  });
  state = workbenchReducer(state, { sourceId: 'video', type: 'setInvocationSource' });
  state = workbenchReducer(state, { destination: 'gallery', type: 'setInvocationDestination' });
  state = workbenchReducer(state, {
    backendSupportsCancellation: true,
    models: [wanModel],
    type: 'submitInvocationSnapshot',
  });

  return state;
};

describe('getLocalQueueItemSource', () => {
  it('reports the video source for an item this client submitted from Video', () => {
    const state = submitVideo();
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
    const localItem = project.queue.items[0]!;

    expect(localItem.snapshot.sourceId).toBe('video');
    expect(getLocalQueueItemSource(state.projects, buildQueueItemOrigin(localItem.id, project.id))).toBe('video');
    // The prefix without a project id is the other shape the submitter emits.
    expect(getLocalQueueItemSource(state.projects, buildQueueItemOrigin(localItem.id))).toBe('video');
  });

  it('reports null — not a source — for an item this client did not submit', () => {
    const state = submitVideo();

    // A foreign item carries no decodable origin, so the caller has to fall back
    // to the Generate-shaped default rather than guessing at video.
    expect(getLocalQueueItemSource(state.projects, 'some-other-client')).toBeNull();
    expect(getLocalQueueItemSource(state.projects, null)).toBeNull();
    expect(getLocalQueueItemSource(state.projects, undefined)).toBeNull();
    // Decodable, but no longer present locally (snapshot cleared).
    expect(getLocalQueueItemSource(state.projects, buildQueueItemOrigin('queue-item-gone'))).toBeNull();
  });
});
