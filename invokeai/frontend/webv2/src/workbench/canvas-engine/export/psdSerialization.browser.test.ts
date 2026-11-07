import { getPsdRequiredBytes, PsdExportController } from '@workbench/canvas-engine/controllers/psdExportController';
import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { readPsd, writePsd, type Psd } from 'ag-psd';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { executePsdExport, planPsdExport, serializePsd } from './psdExport';

const pixels = (side: number): ImageData => {
  const image = new ImageData(side, side);
  let seed = 7;
  for (let i = 0; i < image.data.length; i += 4) {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
    image.data[i] = seed & 255;
    image.data[i + 1] = (seed >>> 8) & 255;
    image.data[i + 2] = (seed >>> 16) & 255;
    image.data[i + 3] = 255;
  }
  return image;
};
const document = (side: number): Psd => ({
  width: side,
  height: side,
  imageData: pixels(side),
  children: [0, 1, 2].map((i) => ({ name: `Layer ${i}`, imageData: pixels(side) })),
});
afterEach(() => vi.unstubAllGlobals());

describe('real PSD serialization lifecycle', () => {
  it('releases workers after failure and repeated success; transfers only owned pixels', async () => {
    const NativeWorker = globalThis.Worker;
    const active = new Set<Worker>();
    const received: unknown[] = [];
    class TrackedWorker extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        const bootstrap = URL.createObjectURL(
          new Blob(
            [
              `self.onmessage = async (event) => {
          await import(${JSON.stringify(new URL(url, location.href).href)});
          if (event.data.psd.imageData) self.postMessage({ id: 0, received: { plain: Object.getPrototypeOf(event.data.psd.imageData) === Object.prototype,
            bytes: event.data.psd.imageData.data.buffer.byteLength,
            first: Array.from(event.data.psd.imageData.data.subarray(0,4)) } });
          self.onmessage(event);
        };`,
            ],
            { type: 'text/javascript' }
          )
        );
        super(bootstrap, options);
        URL.revokeObjectURL(bootstrap);
        this.addEventListener('message', ({ data }) => {
          if (data.id === 0) {
            received.push(data.received);
          }
        });
        active.add(this);
      }
      override terminate(): void {
        super.terminate();
        active.delete(this);
      }
    }
    vi.stubGlobal('Worker', TrackedWorker);
    await expect(serializePsd({ width: 0, height: 1 })).rejects.toThrow('Invalid document size');
    expect(active.size).toBe(0);
    for (let i = 0; i < 2; i += 1) {
      const psd = document(8);
      const expected = new Uint8Array(writePsd(psd));
      const result = serializePsd(psd);
      expect(psd.imageData!.data.byteLength).toBe(0);
      const bytes = await result;
      expect(new Uint8Array(bytes)).toEqual(expected);
      const parsed = readPsd(bytes, { useImageData: true });
      expect(parsed.children).toHaveLength(3);
      expect(Array.from(parsed.imageData!.data.subarray(0, 4))).toEqual([186, 189, 32, 255]);
      expect(active.size).toBe(0);
      expect(received.at(-1)).toEqual({ plain: true, bytes: 8 * 8 * 4, first: [186, 189, 32, 255] });
    }
  });

  it('interrupts an entered production writer, releases its reservation without downloading, and retries', async () => {
    const NativeWorker = globalThis.Worker;
    const active = new Set<Worker>();
    let entered!: () => void;
    const writerEntered = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let finished = false;
    class ObservedWorker extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        // Import the production handler, then observe its synchronous call without delaying or replacing its work.
        const bootstrap = URL.createObjectURL(
          new Blob(
            [
              `self.onmessage = async (event) => {
              await import(${JSON.stringify(new URL(url, location.href).href)});
              const write = self.onmessage;
              self.onmessage = null;
              self.postMessage({ id: 0, phase: 'entered' });
              write.call(self, event);
              self.postMessage({ id: 0, phase: 'finished' });
            };`,
            ],
            { type: 'text/javascript' }
          )
        );
        super(bootstrap, options);
        URL.revokeObjectURL(bootstrap);
        active.add(this);
        this.addEventListener('message', ({ data }) => {
          if (data.id === 0 && data.phase === 'entered') {
            entered();
          }
          if (data.id === 0 && data.phase === 'finished') {
            finished = true;
          }
        });
      }
      override terminate(): void {
        super.terminate();
        active.delete(this);
      }
    }
    vi.stubGlobal('Worker', ObservedWorker);
    const backend = createDomRasterBackend();
    // Enough real pixel/RLE work to keep the writer occupied after its entry message reaches the page.
    const side = 1536;
    const rect = { x: 0, y: 0, width: side, height: side };
    const source = backend.createSurface(side, side);
    source.ctx.putImageData(pixels(side), 0, 0);
    const documentSnapshot = {
      canvas: {
        ...createInitialWorkbenchState().projects[0]!.canvas,
        document: documentFrom([layerContract('layer')]),
      },
      documentGeneration: 0,
    };
    const release = vi.fn();
    const download = vi.fn();
    const memory = new RasterMemoryBudgetController();
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          status: 'ok' as const,
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set<string>(),
            layerSurfaces: new Map([['layer', { rect, surface: source }]]),
            release,
          },
        }),
      getAvailableBytes: () => memory.getAvailableBytes(),
      isDocumentSnapshotCurrent: () => true,
      reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'psd-export' }),
      execute: (plan, fileName, deps) => executePsdExport(plan, fileName, { ...deps, download }),
    });
    try {
      const pending = controller.export('cancelled.psd');
      await Promise.race([
        writerEntered,
        pending.then(() => {
          throw new Error('PSD export settled before the writer entered.');
        }),
      ]);
      expect(finished).toBe(false);
      expect(active.size).toBe(1);
      expect(memory.snapshot().reservedBytes).toBe(0);

      controller.cancel();
      expect(active.size).toBe(0);
      await expect(pending).resolves.toBe('aborted');
      expect(memory.snapshot().reservedBytes).toBe(0);
      expect(release).toHaveBeenCalledOnce();
      expect(download).not.toHaveBeenCalled();
      expect(finished).toBe(false);

      await expect(controller.export('retry.psd')).resolves.toBe('exported');
      expect(download).toHaveBeenCalledOnce();
      expect(release).toHaveBeenCalledTimes(2);
      expect(active.size).toBe(0);
      expect(memory.snapshot().reservedBytes).toBe(0);
    } finally {
      controller.dispose();
      for (const worker of active) {
        worker.terminate();
      }
      source.resize(0, 0);
      memory.dispose();
    }
  });
});

// Informational profile, deliberately outside timing gates. Run with VITE_PSD_PROFILE=1 on a quiet machine.
it.runIf(import.meta.env.VITE_PSD_PROFILE === '1')(
  'profiles medium and large multilayer serialization',
  async ({ annotate, task }) => {
    for (const side of [512, 1024]) {
      const setupStart = performance.now();
      const backend = createDomRasterBackend();
      const source = backend.createSurface(side, side);
      source.ctx.putImageData(pixels(side), 0, 0);
      const sourceSetupMs = performance.now() - setupStart;
      const rect = { x: 0, y: 0, width: side, height: side };
      const plan = planPsdExport(
        [0, 1, 2].map((i) => ({
          id: String(i),
          name: `Layer ${i}`,
          contentRect: rect,
          blendMode: 'normal' as const,
          opacity: 1,
          isEnabled: true,
          transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
        }))
      );
      if (plan.status !== 'ok') {
        throw new Error('Expected an executable profile fixture');
      }
      const pipeline = async (worker: boolean) => {
        let extractionMs = 0;
        let serializationMs = 0;
        let bytes: ArrayBuffer | undefined;
        const start = performance.now();
        await executePsdExport(plan, 'profile.psd', {
          backend,
          getLayerSurface: () => Promise.resolve({ rect, surface: source }),
          readImageData: (surface, region) => {
            const before = performance.now();
            const data = surface.ctx.getImageData(region.x, region.y, region.width, region.height);
            extractionMs += performance.now() - before;
            return data;
          },
          writePsd: async (psd) => {
            const before = performance.now();
            const result = worker ? await serializePsd(psd) : writePsd(psd);
            serializationMs = performance.now() - before;
            return result;
          },
          download: (data) => {
            bytes = data;
          },
        });
        const totalMs = performance.now() - start;
        Object.assign(task.meta, {
          [`pipeline${side}${worker ? 'Worker' : 'Baseline'}`]: {
            sourceSetupMs,
            extractionMs,
            serializationMs,
            totalMs,
            rasterAndCompositionMs: totalMs - extractionMs - serializationMs,
            reservedBytes: getPsdRequiredBytes(plan),
            captureAndDownload: 'fixture-owned source; no live snapshot capture or native download',
          },
        });
        return bytes!;
      };
      const measure = async (
        run: () => Promise<ArrayBuffer>
      ): Promise<{ bytes: ArrayBuffer; ms: number; maxTimerGap: number }> => {
        let last = performance.now();
        let maxTimerGap = 0;
        const timer = setInterval(() => {
          const now = performance.now();
          maxTimerGap = Math.max(maxTimerGap, now - last);
          last = now;
        }, 4);
        const start = performance.now();
        try {
          const bytes = await run();
          const ms = performance.now() - start;
          await new Promise((resolve) => {
            setTimeout(resolve, 20);
          });
          return { bytes, ms, maxTimerGap };
        } finally {
          clearInterval(timer);
        }
      };
      const baseline = await measure(() => pipeline(false));
      const worker = await measure(() => pipeline(true));
      const expected = new Uint8Array(baseline.bytes);
      expect(new Uint8Array(worker.bytes).every((value, i) => value === expected[i])).toBe(true);
      expect(worker.bytes.byteLength).toBe(expected.byteLength);
      const measurement = {
        side,
        layers: 3,
        outputBytes: expected.length,
        baselineMs: baseline.ms,
        baselineMaxTimerGap: baseline.maxTimerGap,
        workerMs: worker.ms,
        workerMaxTimerGap: worker.maxTimerGap,
        peakMemory: 'unavailable: page heap metrics exclude worker and canvas/native allocations',
      };
      Object.assign(task.meta, { [`psd${side}`]: measurement });
      await annotate(JSON.stringify(measurement));
      source.resize(0, 0);
    }
  },
  30_000
);
