import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import { runLayerFilter } from './layerFilterRunner';

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
          return Promise.resolve({ imageName: 'output' });
        },
        uploadIntermediate: (receivedBlob) => {
          expect(receivedBlob).toBe(blob);
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
            return Promise.resolve({ imageName: 'output' });
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
            return Promise.resolve({ imageName: 'output' });
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
        runFilterGraph: () => Promise.resolve({ imageName: 'output' }),
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
