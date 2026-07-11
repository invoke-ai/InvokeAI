import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import { resolveFilterOutputRect, runLayerFilter } from './layerFilterRunner';

describe('resolveFilterOutputRect', () => {
  const source = { height: 60, width: 80, x: 12, y: 24 };

  it.each([
    ['gaussian', 2.1, { height: 74, width: 94, x: 5, y: 17 }],
    ['box', 2.1, { height: 66, width: 86, x: 9, y: 21 }],
  ] as const)('uses exact legacy %s blur padding and actual output dimensions', (blurType, radius, expected) => {
    expect(
      resolveFilterOutputRect({
        filterType: 'img_blur',
        output: { height: expected.height, width: expected.width },
        settings: { blur_type: blurType, radius },
        source,
      })
    ).toEqual(expected);
  });

  it('keeps the legacy Spandrel origin while adopting actual upscale dimensions', () => {
    expect(
      resolveFilterOutputRect({
        filterType: 'spandrel_filter',
        output: { height: 240, width: 320 },
        settings: { scale: 4 },
        source,
      })
    ).toEqual({ height: 240, width: 320, x: 12, y: 24 });
  });

  it.each([
    [Number.NaN, 4],
    [5_000, 12 - 4_096],
  ])('uses the graph-resolved blur radius for malformed or oversized input %s', (radius, expectedX) => {
    expect(
      resolveFilterOutputRect({
        filterType: 'img_blur',
        output: { height: 60, width: 80 },
        settings: { blur_type: 'box', radius },
        source,
      }).x
    ).toBe(expectedX);
  });

  it('retains the source rect for dimension-preserving filters', () => {
    expect(
      resolveFilterOutputRect({
        filterType: 'canny_edge_detection',
        output: { height: 999, width: 999 },
        source,
      })
    ).toEqual(source);
  });
});

describe('runLayerFilter', () => {
  it('encodes, uploads, builds, and runs a filter in order', async () => {
    const surface = createTestStubRasterBackend().createSurface(80, 60);
    const blob = new Blob(['pixels'], { type: 'image/png' });
    const calls: string[] = [];
    const signal = new AbortController().signal;

    const result = await runLayerFilter({
      deps: {
        encodeSurface: (receivedSurface) => {
          expect(receivedSurface).toBe(surface);
          calls.push('encode');
          return Promise.resolve(blob);
        },
        runFilterGraph: ({ graph, outputNodeId, signal: receivedSignal }) => {
          calls.push('run');
          expect(graph.nodes.control_filter).toMatchObject({
            high_threshold: 200,
            image: { image_name: 'input' },
            low_threshold: 100,
            type: 'canny_edge_detection',
          });
          expect(outputNodeId).toBe('control_filter');
          expect(receivedSignal).toBe(signal);
          return Promise.resolve({ height: 60, imageName: 'output', width: 80 });
        },
        uploadIntermediate: (receivedBlob, receivedSignal) => {
          expect(receivedBlob).toBe(blob);
          expect(receivedSignal).toBe(signal);
          calls.push('upload');
          return Promise.resolve({ imageName: 'input' });
        },
      },
      filterType: 'canny_edge_detection',
      input: { rect: { height: 60, width: 80, x: 12, y: 24 }, surface },
      settings: { high_threshold: 200, low_threshold: 100 },
      signal,
    });

    expect(calls).toEqual(['encode', 'upload', 'run']);
    expect(result).toEqual({ height: 60, imageName: 'output', origin: { x: 12, y: 24 }, width: 80 });
  });

  it('rejects a pre-aborted request without invoking dependencies', async () => {
    const controller = new AbortController();
    controller.abort();
    const calls: string[] = [];

    await expect(
      runLayerFilter({
        deps: {
          encodeSurface: () => {
            calls.push('encode');
            return Promise.resolve(new Blob());
          },
          runFilterGraph: () => {
            calls.push('run');
            return Promise.resolve({ height: 1, imageName: 'output', width: 1 });
          },
          uploadIntermediate: () => {
            calls.push('upload');
            return Promise.resolve({ imageName: 'input' });
          },
        },
        filterType: 'canny_edge_detection',
        input: {
          rect: { height: 1, width: 1, x: 0, y: 0 },
          surface: createTestStubRasterBackend().createSurface(1, 1),
        },
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(calls).toEqual([]);
  });

  it('does not build or run the graph after upload aborts the request', async () => {
    const controller = new AbortController();
    const calls: string[] = [];

    await expect(
      runLayerFilter({
        deps: {
          encodeSurface: () => {
            calls.push('encode');
            return Promise.resolve(new Blob());
          },
          runFilterGraph: () => {
            calls.push('run');
            return Promise.resolve({ height: 1, imageName: 'output', width: 1 });
          },
          uploadIntermediate: () => {
            calls.push('upload');
            controller.abort();
            return Promise.resolve({ imageName: 'input' });
          },
        },
        filterType: 'canny_edge_detection',
        input: {
          rect: { height: 1, width: 1, x: 0, y: 0 },
          surface: createTestStubRasterBackend().createSurface(1, 1),
        },
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(calls).toEqual(['encode', 'upload']);
  });

  it('forwards cancellation into an in-flight upload', async () => {
    const controller = new AbortController();
    let uploadSignal: AbortSignal | undefined;

    const pending = runLayerFilter({
      deps: {
        encodeSurface: () => Promise.resolve(new Blob()),
        runFilterGraph: () => Promise.resolve({ height: 1, imageName: 'output', width: 1 }),
        uploadIntermediate: (_blob, signal) => {
          uploadSignal = signal;
          return new Promise((_resolve, reject) => {
            signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
          });
        },
      },
      filterType: 'canny_edge_detection',
      input: {
        rect: { height: 1, width: 1, x: 0, y: 0 },
        surface: createTestStubRasterBackend().createSurface(1, 1),
      },
      signal: controller.signal,
    });
    await Promise.resolve();
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(uploadSignal).toBe(controller.signal);
  });

  it('propagates a filter graph failure', async () => {
    const graphError = new Error('graph failed');

    await expect(
      runLayerFilter({
        deps: {
          encodeSurface: () => Promise.resolve(new Blob()),
          runFilterGraph: () => Promise.reject(graphError),
          uploadIntermediate: () => Promise.resolve({ imageName: 'input' }),
        },
        filterType: 'canny_edge_detection',
        input: {
          rect: { height: 1, width: 1, x: 0, y: 0 },
          surface: createTestStubRasterBackend().createSurface(1, 1),
        },
      })
    ).rejects.toBe(graphError);
  });

  it('preserves a negative layer-local origin', async () => {
    const result = await runLayerFilter({
      deps: {
        encodeSurface: () => Promise.resolve(new Blob()),
        runFilterGraph: () => Promise.resolve({ height: 20, imageName: 'output', width: 30 }),
        uploadIntermediate: () => Promise.resolve({ imageName: 'input' }),
      },
      filterType: 'canny_edge_detection',
      input: {
        rect: { height: 20, width: 30, x: -12, y: -24 },
        surface: createTestStubRasterBackend().createSurface(30, 20),
      },
    });

    expect(result.origin).toEqual({ x: -12, y: -24 });
  });
});
