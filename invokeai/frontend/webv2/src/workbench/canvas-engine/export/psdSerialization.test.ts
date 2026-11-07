import type { Psd } from 'ag-psd';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { serializePsd } from './psdExport';

class TestWorker {
  static instances: TestWorker[] = [];
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: ((event: { preventDefault(): void; message: string }) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  terminate = vi.fn();
  postMessage = vi.fn();
  constructor() {
    TestWorker.instances.push(this);
  }
}
const document = (): Psd => ({
  width: 1,
  height: 1,
  imageData: { width: 1, height: 1, data: new Uint8ClampedArray([1, 2, 3, 255]) },
});
const worker = (): TestWorker => TestWorker.instances.at(-1)!;
afterEach(() => {
  vi.unstubAllGlobals();
  TestWorker.instances = [];
});

describe('operation-owned PSD serialization worker', () => {
  it('transfers inputs, ignores another request, and cleans up on success', async () => {
    vi.stubGlobal('Worker', TestWorker);
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, 'removeEventListener');
    const psd = document();
    const result = serializePsd(psd, controller.signal);
    expect(worker().postMessage).toHaveBeenCalledWith({ id: 1, psd }, [psd.imageData!.data.buffer]);
    worker().onmessage!({ data: { id: 2, bytes: new ArrayBuffer(3) } });
    expect(worker().terminate).not.toHaveBeenCalled();
    const bytes = new ArrayBuffer(7);
    worker().onmessage!({ data: { id: 1, bytes } });
    await expect(result).resolves.toBe(bytes);
    expect(worker().terminate).toHaveBeenCalledOnce();
    expect(worker().onmessage).toBeNull();
    expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
  });
  it('terminates on cancellation and can retry without stale completion', async () => {
    vi.stubGlobal('Worker', TestWorker);
    const controller = new AbortController();
    const result = serializePsd(document(), controller.signal);
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(worker().terminate).toHaveBeenCalledOnce();
    expect(worker().onmessage).toBeNull();
    const retry = serializePsd(document());
    worker().onmessage!({ data: { id: 1, bytes: new ArrayBuffer(1) } });
    await expect(retry).resolves.toHaveProperty('byteLength', 1);
    expect(TestWorker.instances).toHaveLength(2);
  });
  it.each(['error', 'messageerror', 'writer', 'transfer'] as const)('cleans up after %s failure', async (kind) => {
    vi.stubGlobal('Worker', TestWorker);
    const psd = document();
    if (kind === 'transfer') {
      Object.defineProperty(psd, 'imageData', {
        get: () => {
          throw new Error('transfer failed');
        },
      });
    }
    const result = serializePsd(psd);
    if (kind === 'error') {
      worker().onerror!({ message: 'load failed', preventDefault: vi.fn() });
    }
    if (kind === 'messageerror') {
      worker().onmessageerror!();
    }
    if (kind === 'writer') {
      worker().onmessage!({ data: { id: 1, error: 'writer failed' } });
    }
    await expect(result).rejects.toThrow();
    expect(worker().terminate).toHaveBeenCalledOnce();
    expect(worker().onerror).toBeNull();
    expect(worker().onmessageerror).toBeNull();
  });
  it('does not allocate for an already aborted operation', async () => {
    vi.stubGlobal('Worker', TestWorker);
    await expect(serializePsd(document(), AbortSignal.abort())).rejects.toMatchObject({ name: 'AbortError' });
    expect(TestWorker.instances).toHaveLength(0);
  });
});

it('reports request deserialization failure through the private worker protocol', async () => {
  const scope = { postMessage: vi.fn(), onmessage: null, onmessageerror: null as null | (() => void) };
  vi.stubGlobal('self', scope);
  await import('./psdSerialization.worker');
  scope.onmessageerror!();
  expect(scope.postMessage).toHaveBeenCalledWith({ id: 1, error: 'PSD serialization request could not be read.' });
});
