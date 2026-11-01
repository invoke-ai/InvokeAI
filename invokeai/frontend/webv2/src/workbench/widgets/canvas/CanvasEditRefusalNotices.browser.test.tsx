/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { CanvasEditRefusal, CanvasLayerEditRefusal, CanvasNodeContract } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { getDocumentIndex, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { groupContract, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { CanvasEditRefusalNotices } from './CanvasEditRefusalNotices';

await i18next.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => {
    toaster.remove();
    root?.unmount();
  });
  host?.remove();
  host = null;
  root = null;
});

const openToasts = (): HTMLElement[] =>
  [...document.querySelectorAll<HTMLElement>('[data-part="root"][data-scope="toast"]')].filter(
    (toast) => toast.getAttribute('data-state') === 'open'
  );

const toastTexts = () =>
  openToasts().map((toast) => ({
    actions: [...toast.querySelectorAll('button')]
      .map((button) => button.textContent)
      .filter((label) => label !== '' && label !== null),
    message: toast.querySelector('[data-part="description"]')?.textContent,
    title: toast.querySelector('[data-part="title"]')?.textContent,
  }));

/** The notices over a real structural engine; `emit` plays a refusal the way the engine's tools report one. */
const mount = async (layers: readonly CanvasNodeContract[], selectedLayerId: string | null) => {
  const stub = createStructuralEngineStub({ layers, selectedLayerId });
  const listeners = new Set<(refusal: CanvasEditRefusal) => void>();
  const engine = {
    ...stub.engine,
    tools: {
      onEditRefused: (listener: (refusal: CanvasEditRefusal) => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <AppToaster />
        <CanvasEditRefusalNotices engine={engine as never} />
      </ChakraProvider>
    )
  );
  const emit = (refusal: CanvasLayerEditRefusal) => act(() => listeners.forEach((listener) => listener(refusal)));
  return { emit, stub };
};

const refusal = (
  status: CanvasLayerEditRefusal['status'],
  layerId: string,
  layerName: string,
  canPaintNewLayer = false
): CanvasLayerEditRefusal => ({ canPaintNewLayer, layerId, layerName, status });

describe('CanvasEditRefusalNotices', () => {
  it.each([
    ['layer-hidden', '“Sky” is hidden', 'Show the layer to edit it.'],
    ['layer-locked', '“Sky” is locked', 'Unlock the layer to edit it.'],
    [
      'layer-alpha-locked',
      '“Sky” has its transparency locked',
      'The eraser can’t remove pixels while transparency is locked.',
    ],
    ['layer-loading', '“Sky” is still loading', 'Its pixels are on their way. Try again in a moment.'],
  ] as const)('explains a %s refusal with no offer', async (status, title, message) => {
    const { emit } = await mount([layerContract('sky', 'raster', { name: 'Sky' })], 'sky');
    await emit(refusal(status, 'sky', 'Sky'));
    await expect.poll(toastTexts).toEqual([{ actions: [], message, title }]);
  });

  it('replaces a repeated refusal without an offer, one toast per layer and reason', async () => {
    const { emit } = await mount(
      [layerContract('sky', 'raster', { name: 'Sky' }), layerContract('sea', 'raster', { name: 'Sea' })],
      'sky'
    );
    for (let press = 0; press < 3; press += 1) {
      await emit(refusal('layer-alpha-locked', 'sky', 'Sky'));
    }
    await emit(refusal('layer-locked', 'sky', 'Sky'));
    await emit(refusal('layer-alpha-locked', 'sea', 'Sea'));
    await emit(refusal('layer-alpha-locked', 'sea', 'Sea'));

    await expect
      .poll(() => toastTexts().map(({ title }) => title))
      .toEqual(['“Sea” has its transparency locked', '“Sky” is locked', '“Sky” has its transparency locked']);
  });

  it('offers a new layer above the refused one, not the selection, selects it, and closes the other offers', async () => {
    const { emit, stub } = await mount(
      [layerContract('top', 'raster', { name: 'Top' }), layerContract('photo', 'raster', { name: 'Photo' })],
      'top'
    );
    await emit(refusal('layer-locked', 'photo', 'Photo', true));
    await emit(refusal('layer-locked', 'photo', 'Photo', true));
    await emit(refusal('layer-hidden', 'top', 'Top', true));
    // A repeated press replaces its layer's offer.
    await expect.poll(() => toastTexts().map(({ title }) => title)).toEqual(['“Top” is hidden', '“Photo” is locked']);
    expect(toastTexts()[1]).toEqual({
      actions: ['Paint on a new layer'],
      message: 'Unlock the layer to paint on it, or paint on a new layer above it.',
      title: '“Photo” is locked',
    });

    const photoOffer = openToasts().find((toast) => toast.textContent?.includes('“Photo” is locked'))!;
    await userEvent.click(
      [...photoOffer.querySelectorAll('button')].find((button) => button.textContent === 'Paint on a new layer')!
    );

    const ids = () => getDocumentLeaves(stub.document()).map((layer) => layer.id);
    await expect.poll(ids).toHaveLength(3);
    const added = ids().find((id) => id !== 'top' && id !== 'photo')!;
    expect(ids()).toEqual(['top', added, 'photo']);
    expect(stub.document().selectedLayerId).toBe(added);
    await expect.poll(openToasts).toHaveLength(0);
  });

  it('puts the new layer above the outermost group that locks the refused layer', async () => {
    const child = layerContract('child', 'raster', { name: 'Child' });
    const { emit, stub } = await mount(
      [groupContract('outer', [groupContract('inner', [child], { isLocked: true })], { isLocked: true })],
      'child'
    );
    await emit(refusal('layer-locked', 'child', 'Child', true));
    await userEvent.click(page.getByRole('button', { name: 'Paint on a new layer' }));

    await expect.poll(() => getDocumentLeaves(stub.document())).toHaveLength(2);
    const index = getDocumentIndex(stub.document());
    const added = getDocumentLeaves(stub.document()).find((layer) => layer.id !== 'child')!;
    expect(index.byId.get(added.id)?.path).toEqual([]);
    expect(index.nodes.map(({ node }) => node.id)).toEqual([added.id, 'outer', 'inner', 'child']);
  });
});
