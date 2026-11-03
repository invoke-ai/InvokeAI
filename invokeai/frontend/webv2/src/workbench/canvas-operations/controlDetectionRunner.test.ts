import { describe, expect, it, vi } from 'vitest';

import { ControlDetectionFailure, runControlDetection } from './controlDetectionRunner';
import { FILTER_NODE_ID } from './filterGraphs';

const canny = { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' };
const output = (width: number, height: number) => ({ height, imageName: 'map.png', width });

describe('runControlDetection', () => {
  it('builds the graph on the persisted image at its content size and keeps the map in its frame', async () => {
    const runFilterGraph = vi.fn(() => Promise.resolve(output(30, 20)));
    const result = await runControlDetection({
      detection: canny,
      input: { bitmap: { height: 20, imageName: 'paint.png', width: 30 }, offset: { x: -4, y: 6 }, type: 'paint' },
      runFilterGraph,
      signal: new AbortController().signal,
    });

    const [{ graph, outputNodeId }] = runFilterGraph.mock.calls[0] as unknown as [
      { graph: { nodes: Record<string, Record<string, unknown>> }; outputNodeId: string },
    ];
    // Processed over a white input-sized background, with the input's alpha restored on the output.
    expect(graph.nodes[FILTER_NODE_ID]).toMatchObject({ high_threshold: 200, type: 'canny_edge_detection' });
    expect(graph.nodes.control_filter_background).toMatchObject({ height: 20, width: 30 });
    expect(graph.nodes.control_filter_flatten).toMatchObject({ image: { image_name: 'paint.png' } });
    expect(graph.nodes.control_filter_source_alpha).toMatchObject({ image: { image_name: 'paint.png' } });
    expect(outputNodeId).toBe('control_filter_restore_alpha');
    expect(result).toEqual({
      image: { height: 20, imageName: 'map.png', width: 30 },
      rect: { height: 20, width: 30, x: -4, y: 6 },
    });
  });

  it('places an image input at its origin', async () => {
    const result = await runControlDetection({
      detection: canny,
      input: { image: { height: 48, imageName: 'photo.png', width: 64 }, type: 'image' },
      runFilterGraph: () => Promise.resolve(output(64, 48)),
      signal: new AbortController().signal,
    });

    expect(result.rect).toEqual({ height: 48, width: 64, x: 0, y: 0 });
  });

  it.each([
    [
      'unknown-detector',
      { detection: { settings: {}, type: 'retired_detector' }, output: output(64, 64), image: true },
    ],
    ['source-not-ready', { detection: canny, output: output(64, 64), image: false }],
    ['output-dimensions', { detection: canny, output: output(32, 32), image: true }],
  ] as const)('reports %s', async (code, scenario) => {
    const runFilterGraph = vi.fn(() => Promise.resolve(scenario.output));
    const run = runControlDetection({
      detection: scenario.detection,
      input: scenario.image
        ? { image: { height: 64, imageName: 'photo.png', width: 64 }, type: 'image' }
        : { bitmap: null, type: 'paint' },
      runFilterGraph,
      signal: new AbortController().signal,
    });

    await expect(run).rejects.toBeInstanceOf(ControlDetectionFailure);
    await expect(run).rejects.toMatchObject({ code });
  });

  it('wraps graph failures with their message and lets aborts through', async () => {
    const input = { image: { height: 64, imageName: 'photo.png', width: 64 }, type: 'image' } as const;
    await expect(
      runControlDetection({
        detection: canny,
        input,
        runFilterGraph: () => Promise.reject(new Error('queue offline')),
        signal: new AbortController().signal,
      })
    ).rejects.toMatchObject({ code: 'process-failed', detail: 'queue offline' });

    const controller = new AbortController();
    const run = runControlDetection({
      detection: canny,
      input,
      runFilterGraph: () => {
        controller.abort();
        return Promise.reject(new Error('cancelled'));
      },
      signal: controller.signal,
    });
    await expect(run).rejects.toMatchObject({ name: 'AbortError' });
  });
});
