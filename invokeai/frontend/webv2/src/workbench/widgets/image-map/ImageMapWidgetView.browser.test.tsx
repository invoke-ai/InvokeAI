import type { ImageMapState } from '@workbench/image-map/api';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@workbench/WorkbenchContext', () => ({
  useWidgetValuesSelector: () => false,
}));

// The real one pulls the ~1.5MB plotly chunk; the badge under test is its
// sibling, not its child, so a stand-in is enough.
vi.mock('./ImageMapPlot', () => ({ default: () => <div data-testid="plot" /> }));

vi.mock('@workbench/image-map/imageMapStore', async (importOriginal) => {
  const original = (await importOriginal()) as object;

  return {
    ...original,
    ensureImageMapLoaded: vi.fn(),
    refreshImageIndexStatus: vi.fn(),
    refreshImageMapPoints: vi.fn(),
    setClusterLabelsEnabled: vi.fn(),
  };
});

import { imageMapStore } from '@workbench/image-map/imageMapStore';

import { ImageMapWidgetView } from './ImageMapWidgetView';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const dataFor = (state: Extract<ImageMapState, 'disabled' | 'model_missing'>) => ({
  clusterEps: null,
  modelName: state === 'model_missing' ? 'clip-vit-large-patch14' : null,
  pointCount: 0,
  points: [],
  stale: false,
  state,
  updatedAt: null,
  visibleHash: null,
});

const renderState = async (state: Extract<ImageMapState, 'disabled' | 'model_missing'>) => {
  imageMapStore.setSnapshot({
    clusterLabels: null,
    clusterLabelsHash: null,
    data: dataFor(state),
    error: null,
    indexCounts: null,
    indexUpdatedAt: null,
    loadState: 'loaded',
    renderError: null,
  });

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ImageMapWidgetView {...({} as WidgetViewProps)} />
      </ChakraProvider>
    )
  );
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('Image Map unavailable states', () => {
  it.each([
    ['model_missing', 'Embedding model not installed'],
    ['disabled', 'Image indexing is off'],
  ] as const)('keeps the %s diagnosis visible while exposing a failed refresh', async (state, title) => {
    await renderState(state);

    expect(host?.textContent).toContain(title);
    expect(host?.querySelector('button')).toBeNull();

    await act(() => {
      imageMapStore.patchSnapshot({ error: 'The server could not be reached.', loadState: 'error' });
    });

    expect(host?.textContent).toContain(title);
    expect(host?.textContent).toContain('The server could not be reached.');
    expect(host?.querySelector('[role="alert"]')?.textContent).toBe('The server could not be reached.');
    expect(host?.querySelector('button')?.textContent).toBe('Retry');
  });
});

describe('Image Map indexing activity', () => {
  const renderMapWithCounts = async (
    counts: { total: number; embedded: number; pending: number; failed: number } | null
  ) => {
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsHash: null,
      data: {
        clusterEps: null,
        modelName: null,
        pointCount: 2,
        points: [
          { cluster: 0, imageName: 'a.png', x: 0, y: 0 },
          { cluster: 0, imageName: 'b.png', x: 1, y: 1 },
        ],
        stale: false,
        state: 'ready',
        updatedAt: '2026-08-24T01:00:00',
        visibleHash: 'hash',
      },
      error: null,
      indexCounts: counts,
      indexUpdatedAt: counts ? Date.now() : null,
      loadState: 'loaded',
      renderError: null,
    });

    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <ImageMapWidgetView {...({} as WidgetViewProps)} />
        </ChakraProvider>
      )
    );
    // The plot is `lazy()`, so the first render in the file waits on the
    // dynamic import and then on the re-render Suspense schedules once it
    // resolves. Polled rather than flushed a fixed number of times: the badge
    // has to be asserted against the resolved tree, not the fallback.
    for (let attempt = 0; attempt < 50 && !host?.querySelector('[data-testid="plot"]'); attempt += 1) {
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });
    }
  };

  it('reports an index run over the map instead of drawing it silently', async () => {
    // The has-points branch preempts the progress panel, which is right — a
    // usable stale map beats a progress bar — but it used to do so with no
    // sign that anything was happening, which is what a model-change re-index
    // looks like from the panel: the old map, no labels, no explanation.
    await renderMapWithCounts({ embedded: 1204, failed: 0, pending: 16846, total: 18050 });

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).toContain('indexing 1,204/18,050');
    // The map stays: the badge must not replace it.
    expect(host?.textContent).not.toContain('Indexing images');
  });

  it('names the labels in the badge, since they vanish while the vocabulary rebuilds', async () => {
    await renderMapWithCounts({ embedded: 1204, failed: 0, pending: 16846, total: 18050 });

    const progressbar = host?.querySelector('[role="progressbar"]');

    expect(progressbar?.getAttribute('aria-label')).toContain('cluster labels update as images finish');
  });

  it('shows no badge once the index is idle', async () => {
    await renderMapWithCounts({ embedded: 18050, failed: 0, pending: 0, total: 18050 });

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).not.toContain('indexing');
  });

  it('shows no badge when the counts are absent, as for a non-admin', async () => {
    await renderMapWithCounts(null);

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).not.toContain('indexing');
  });
});
