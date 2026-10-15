import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('@features/workflow/ui/WorkflowUiContext', async () => {
  const { DEFAULT_THEME_ID } = await import('@theme/themes');
  return {
    useWorkflowPreferencesSelector: (selector: (preferences: { themeId: string }) => unknown) =>
      selector({ themeId: DEFAULT_THEME_ID }),
  };
});
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const { GraphPreviewSnapshot } = await import('./GraphPreviewSnapshot');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const GRAPH: WorkflowPreviewGraph = {
  edges: [
    { id: 'e1', sourceField: 'value', sourceNodeId: 'seed', targetField: 'seed', targetNodeId: 'noise' },
    { id: 'e2', sourceField: 'noise', sourceNodeId: 'noise', targetField: 'noise', targetNodeId: 'denoise' },
  ],
  id: 'snapshot-graph',
  nodes: [
    { id: 'seed', inputs: { value: 1 }, type: 'integer' },
    { id: 'noise', inputs: {}, type: 'noise' },
    { id: 'denoise', inputs: { steps: 30 }, type: 'denoise_latents' },
  ],
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Pixels that differ from the corner pixel, i.e. anything painted over the background. */
const countPaintedPixels = async (image: Blob): Promise<{ height: number; painted: number; width: number }> => {
  const bitmap = await createImageBitmap(image);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d')!;
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  const [r, g, b] = [data[0], data[1], data[2]];
  let painted = 0;

  for (let index = 0; index < data.length; index += 4) {
    if (Math.abs(data[index]! - r!) + Math.abs(data[index + 1]! - g!) + Math.abs(data[index + 2]! - b!) > 24) {
      painted += 1;
    }
  }

  return { height: bitmap.height, painted, width: bitmap.width };
};

describe('GraphPreviewSnapshot', () => {
  it('captures the rendered graph, not just its background, at the thumbnail shape', async () => {
    const onCapture = vi.fn();
    const onError = vi.fn();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <GraphPreviewSnapshot graph={GRAPH} onCapture={onCapture} onError={onError} />
        </ChakraProvider>
      );
    });

    await vi.waitFor(() => expect(onCapture.mock.calls.length + onError.mock.calls.length).toBe(1), {
      timeout: 15_000,
    });
    expect(onError).not.toHaveBeenCalled();

    const image = onCapture.mock.calls[0]![0] as Blob;
    expect(image.type).toBe('image/png');
    const { height, painted, width } = await countPaintedPixels(image);
    expect([width, height]).toEqual([960, 640]);
    // Three node cards and their edges cover well over a percent of the frame.
    expect(painted).toBeGreaterThan(width * height * 0.01);
  });
});
