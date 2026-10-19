import type { GenerateWidgetValues } from '@features/generation/contracts';
import type * as ModelsModule from '@features/models';
import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { ImageRecallKind } from '@workbench/image-actions';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { createDefaultVideoWidgetValues } from '@features/video';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { LocalRecallSnapshot } from './useLocalRecallSnapshot';

import { useQueueItemRecall } from './useQueueItemRecall';

const mocks = vi.hoisted(() => ({
  notifyInfo: vi.fn(),
  notifySuccess: vi.fn(),
  openWidget: vi.fn(),
  patchValues: vi.fn(),
  setSettings: vi.fn(),
  snapshot: undefined as LocalRecallSnapshot | null | undefined,
}));

vi.mock('./useLocalRecallSnapshot', () => ({ useLocalRecallSnapshot: () => mocks.snapshot }));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWidgetValuesSelector: () => ({}),
  useWorkbenchCommands: () => ({
    generation: { setSettings: mocks.setSettings },
    widgets: { patchValues: mocks.patchValues },
  }),
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsModule>()),
  ensureModelsLoaded: () => Promise.resolve(),
  useModelsSelector: () => [],
}));
vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: mocks.notifyInfo, success: mocks.notifySuccess }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => mocks.openWidget }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

seedArchitectureCapabilities();

const NO_META: QueueGenerationMeta = {};
const generateSnapshot = {
  positivePrompt: 'snapshot prompt',
  seed: 7,
  seedMode: 'fixed',
} as GenerateWidgetValues;

type Handle = ReturnType<typeof useQueueItemRecall>;
const handle = createRef<Handle>();
const Probe = ({ meta, ref }: { meta: QueueGenerationMeta; ref: Ref<Handle> }) => {
  const value = useQueueItemRecall('webv2:p:project-1:q:item-1', meta);
  useImperativeHandle(ref, () => value, [value]);
  return null;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (snapshot: LocalRecallSnapshot | null | undefined, meta: QueueGenerationMeta = NO_META) => {
  mocks.snapshot = snapshot;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root?.render(<Probe meta={meta} ref={handle} />));
};
const recall = (kind: ImageRecallKind) => act(() => handle.current!.recall(kind));

beforeEach(() => vi.clearAllMocks());
afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useQueueItemRecall', () => {
  it('offers nothing and stays inert while the durable snapshot lookup is pending', async () => {
    await render(undefined);
    expect(Object.values(handle.current!.capabilities).some(Boolean)).toBe(false);
    await recall('all');
    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).not.toHaveBeenCalled();
  });

  it('recalls a generate submission into the Generate panel and reveals it', async () => {
    await render({ generateValues: generateSnapshot, sourceId: 'canvas' });
    expect(handle.current!.capabilities).toMatchObject({ all: true, prompts: true, seed: true });

    await recall('all');

    expect(mocks.setSettings).toHaveBeenCalledWith(generateSnapshot);
    expect(mocks.openWidget).toHaveBeenCalledWith('generate', { preferredRegions: ['left'] });
    expect(mocks.notifySuccess).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.settingsRecalledDescription');
  });

  it('patches a video submission into the Video panel instead', async () => {
    const videoValues = { ...createDefaultVideoWidgetValues(), positivePrompt: 'video prompt' };
    await render({ sourceId: 'video', videoValues });

    await recall('all');

    expect(mocks.patchValues).toHaveBeenCalledWith('video', videoValues);
    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.openWidget).toHaveBeenCalledWith('video', { preferredRegions: ['left'] });
  });

  it('explains an unavailable recall instead of writing anything', async () => {
    // A foreign item: no snapshot, only the executed prompt from the session.
    await render(null, { positivePrompt: 'session prompt' });
    expect(handle.current!.capabilities).toMatchObject({ all: false, prompts: true, seed: false });

    await recall('seed');

    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.recallUnavailable');
  });

  it('names the missing capability table, not a missing model, while the table is absent', async () => {
    await render(null, { positivePrompt: 'session prompt' });

    await act(() => resetArchitectureCapabilities());
    try {
      await recall('seed');
    } finally {
      await act(() => setArchitectureCapabilities(architectureCapabilitiesFixture));
    }

    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.recallUnavailableCapabilities');
  });
});
