import type { ImageMapPoint } from '@workbench/image-map/api';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { ImageMapWidgetFooter } from './ImageMapWidgetFooter';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const points = (...clusters: number[]): ImageMapPoint[] =>
  clusters.map((cluster, index) => ({
    cluster,
    item: { kind: 'image', name: `${index}.png` },
    key: `image:${index}.png` as ImageMapPoint['key'],
    x: index,
    y: index,
  }));

/** The narrowest a floating window gets (`workbench/floatingWindows.ts`). */
const NARROW_HOST = '280px';

/** Every other segment the row can show at once, to squeeze the readout hardest. */
const CROWDED = { embedded: 123456, failed: 12, pending: 60544, total: 184012 };

const renderFooter = async (clusters: number[], width?: string, crowded = false) => {
  imageMapStore.setSnapshot({
    clusterLabels: null,
    clusterLabelsEps: null,
    clusterLabelsHash: null,
    data: {
      clusterEps: 0.2,
      modelName: null,
      pointCount: clusters.length,
      points: points(...clusters),
      stale: crowded,
      state: 'ready',
      updatedAt: null,
      visibleHash: null,
    },
    error: null,
    indexCounts: crowded ? CROWDED : null,
    indexUpdatedAt: crowded ? Date.now() : null,
    loadState: 'loaded',
    renderError: null,
  });

  if (width && host) {
    host.style.width = width;
  }

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ImageMapWidgetFooter {...({} as WidgetViewProps)} />
      </ChakraProvider>
    )
  );
};

const readout = (): HTMLElement => {
  const found = Array.from(host?.querySelectorAll('p, span, div') ?? []).find((element) =>
    element.textContent?.startsWith('· Cluster count:')
  );

  if (!(found instanceof HTMLElement)) {
    throw new Error('the cluster readout is not in the document');
  }

  return found;
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

describe('ImageMapWidgetFooter', () => {
  it('reports the clustering beside the point count', async () => {
    await renderFooter([0, 0, 0, 1, 1, -1]);

    expect(host?.textContent).toContain('6 points');
    expect(host?.textContent).toContain(
      'Cluster count: 2, Largest cluster: 3 media points, Unclustered: 1 media point'
    );
  });

  it('stays readable in a narrow widget, where the text itself cannot be', async () => {
    // This segment is the only shrinkable item in the row, so it absorbs all
    // the overflow: at 280px the numbers are clipped out of the page and the
    // tooltip is the only way left to read them.
    // Crowded: stale, indexing and skipped all showing, so every other
    // segment is competing for the row. They are all nowrap and refuse to
    // shrink, which is what makes this one absorb the whole overflow.
    await renderFooter([0, 0, 0, 1, 1, -1], NARROW_HOST, true);

    const element = readout();

    expect(host?.scrollWidth).toBeLessThanOrEqual((host?.clientWidth ?? 0) + 1);
    expect(element.scrollWidth).toBeGreaterThan(element.clientWidth);
    // What is left of the readout is the tooltip's only hover target, so the
    // row has to leave it a usable one. It survives on flex leftover alone —
    // every sibling refuses to shrink — which is what this pins: a sibling
    // that starts growing greedily would take the target away.
    expect(element.clientWidth).toBeGreaterThanOrEqual(40);

    await act(() => {
      element.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' }));
      element.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true, pointerType: 'mouse' }));
    });
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 500);
    });

    const tooltip = document.querySelector('[role="tooltip"]');

    expect(tooltip?.textContent).toBe('Cluster count: 2, Largest cluster: 3 media points, Unclustered: 1 media point');
  });

  it('says so when nothing clustered', async () => {
    // The symptom being diagnosed: a drawn map where every point is noise.
    await renderFooter([-1, -1, -1]);

    expect(host?.textContent).toContain(
      'Cluster count: 0, Largest cluster: 0 media points, Unclustered: 3 media points'
    );
  });
});
