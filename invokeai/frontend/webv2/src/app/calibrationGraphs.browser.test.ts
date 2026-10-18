import type { ModelConfig, WorkingMemoryCase } from '@features/models';

import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { afterEach, describe, expect, it, vi } from 'vitest';

/** Serves the capability table; module mocks cannot be used here, since they would instrument the worker's modules. */
const fetchCapabilities = () =>
  vi
    .spyOn(globalThis, 'fetch')
    .mockImplementation(() => Promise.resolve(Response.json(architectureCapabilitiesFixture)));

const SDXL = {
  base: 'sdxl',
  format: 'diffusers',
  hash: 'blake3:sdxl',
  key: 'sdxl',
  name: 'SDXL',
  type: 'main',
  variant: 'normal',
} as ModelConfig;

const planCase = (id: string, modelKey = 'sdxl'): WorkingMemoryCase => ({
  components: {},
  estimator: 'sdxl_vae_decode',
  height: 768,
  id,
  line: 'decode_fp32',
  mode: 'txt2img',
  model_key: modelKey,
  negative_prompt: '',
  positive_prompt: 'a lake',
  seed: 1,
  settings: { vaePrecision: 'fp32' },
  steps: 2,
  width: 768,
});

afterEach(() => {
  setModelsSnapshotForTests({ models: [], status: 'idle' });
  vi.restoreAllMocks();
});

/** Runs the real worker: its bundle must start in a browser and compile without the page's modules. */
describe('compileCalibrationGraphs', () => {
  it('compiles the cases in a worker over the tab’s models and the served capability table', async () => {
    const fetch = fetchCapabilities();
    setModelsSnapshotForTests({ models: [SDXL], status: 'loaded' });
    const { compileCalibrationGraphs } = await import('./calibrationGraphs');

    const [compiled] = await compileCalibrationGraphs([planCase('fp32-768')], new AbortController().signal);
    const nodes = Object.values((compiled!.graph as { nodes: Record<string, Record<string, unknown>> }).nodes);

    expect(String(fetch.mock.calls[0]?.[0])).toMatch(/\/api\/v2\/models\/capabilities$/);
    expect(compiled!.caseId).toBe('fp32-768');
    expect(nodes.find((node) => node.type === 'l2i')).toMatchObject({ fp32: true });
    expect(nodes.find((node) => node.type === 'noise')).toMatchObject({ height: 768, width: 768 });
  });

  it('rejects with the compiler’s message for a case that cannot compile', async () => {
    fetchCapabilities();
    setModelsSnapshotForTests({ models: [SDXL], status: 'loaded' });
    const { compileCalibrationGraphs } = await import('./calibrationGraphs');

    await expect(
      compileCalibrationGraphs([planCase('lost', 'removed-model')], new AbortController().signal)
    ).rejects.toThrow(/^Case lost: model_key "removed-model" is not a selectable main model/);
  });

  /** A worker that never answers: its request is dropped. Resolves once the request would have been sent. */
  const silenceWorker = () => {
    const terminate = vi.spyOn(Worker.prototype, 'terminate');
    let sent = () => undefined as void;
    const posted = new Promise<void>((resolve) => {
      sent = resolve;
    });
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(() => sent());

    return { posted, terminate };
  };

  it('stops the worker and rejects when its owner aborts', async () => {
    fetchCapabilities();
    setModelsSnapshotForTests({ models: [SDXL], status: 'loaded' });
    const { compileCalibrationGraphs } = await import('./calibrationGraphs');
    const { posted, terminate } = silenceWorker();
    const owner = new AbortController();

    const compiling = compileCalibrationGraphs([planCase('fp32-768')], owner.signal);
    await posted;
    owner.abort(new Error('Calibration stopped.'));

    await expect(compiling).rejects.toThrow('Calibration stopped.');
    expect(terminate).toHaveBeenCalledTimes(1);
  });

  it('stops a worker that does not answer in time and rejects', async () => {
    fetchCapabilities();
    setModelsSnapshotForTests({ models: [SDXL], status: 'loaded' });
    const { COMPILE_TIMEOUT_MS, compileCalibrationGraphs } = await import('./calibrationGraphs');
    const { posted, terminate } = silenceWorker();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

    try {
      const compiling = compileCalibrationGraphs([planCase('fp32-768')], new AbortController().signal);
      const outcome = expect(compiling).rejects.toThrow(/did not answer within 60 s/);
      await posted;
      vi.advanceTimersByTime(COMPILE_TIMEOUT_MS - 1);
      expect(terminate).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);

      await outcome;
      expect(terminate).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
