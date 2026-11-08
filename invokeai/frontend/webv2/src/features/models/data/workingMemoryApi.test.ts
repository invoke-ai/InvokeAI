import { expect, it, vi } from 'vitest';

import type { ModelsDataPort } from './transport';

const requestJson = vi.hoisted(() => vi.fn((_path: string, _init?: RequestInit) => Promise.resolve({})));

vi.mock('./transport', () => ({
  browserModelsDataPort: {
    buildUrl: (path: string) => path,
    request: vi.fn(),
    requestJson: requestJson as unknown as ModelsDataPort['requestJson'],
  } satisfies ModelsDataPort,
}));

const { calibrateWorkingMemory } = await import('./workingMemoryApi');

it('sends the attempt id with its step count and the compiled model identities, in the server field names', async () => {
  await calibrateWorkingMemory('sdxl main', {
    attemptId: 'attempt-1',
    attemptSteps: 18,
    data: [],
    graph: { id: 'case' },
    graphModels: { sdxl: { base: 'sdxl', format: 'diffusers', hash: 'blake3:sdxl', type: 'main', variant: null } },
    idempotencyKey: 'key-0',
  });

  const [path, init] = requestJson.mock.calls[0]!;
  expect(path).toBe('/api/v2/models/i/sdxl%20main/working_memory/calibrate');
  expect(init?.method).toBe('POST');
  expect(JSON.parse(init?.body as string)).toEqual({
    attempt_id: 'attempt-1',
    attempt_steps: 18,
    data: [],
    graph: { id: 'case' },
    graph_models: { sdxl: { base: 'sdxl', format: 'diffusers', hash: 'blake3:sdxl', type: 'main', variant: null } },
    idempotency_key: 'key-0',
  });
});
