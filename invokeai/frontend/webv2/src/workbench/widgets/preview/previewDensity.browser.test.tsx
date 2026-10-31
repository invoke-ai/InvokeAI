import type { WorkbenchRegion } from '@workbench/widgetContracts';

import { act, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { usePreviewDensity } from './previewDensity';
import { previewStageStore, usePreviewStageContext } from './previewHeaderStore';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const published: { current: string | null } = { current: null };

/** The view's wiring: its measured root publishes to the stage store the header reads. */
const View = ({ region, widthPx }: { region: WorkbenchRegion; widthPx: number }) => {
  const { rootRef } = usePreviewDensity(region, previewStageStore.setDensity);

  return <div ref={rootRef} style={{ width: `${widthPx}px` }} />;
};

const HeaderProbe = () => {
  const { density } = usePreviewStageContext();
  const ref = useCallback(() => {
    published.current = density;
  }, [density]);

  return <span ref={ref} />;
};

const settle = (action: () => void) =>
  act(async () => {
    action();
    await Promise.resolve();
  });

const render = async (region: WorkbenchRegion, widthPx: number) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root?.render(
      <>
        <View region={region} widthPx={widthPx} />
        <HeaderProbe />
      </>
    )
  );
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  previewStageStore.clear();
});

describe('Preview density publishing', () => {
  it('sizes the header by measured width: a narrow center is compact, a wide floating window full', async () => {
    await render('center', 500);
    await vi.waitFor(() => expect(published.current).toBe('compact'));
    await settle(() => root?.unmount());

    await render('floating', 640);
    await vi.waitFor(() => expect(published.current).toBe('full'));
  });

  it('stops publishing when the view goes away, so the header falls back to compact', async () => {
    await render('center', 900);
    await vi.waitFor(() => expect(published.current).toBe('full'));

    // Unmount the view alone; the header probe stays to read what is left.
    await settle(() => root?.render(<HeaderProbe />));
    await vi.waitFor(() => expect(published.current).toBeNull());
  });
});
