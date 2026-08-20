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
