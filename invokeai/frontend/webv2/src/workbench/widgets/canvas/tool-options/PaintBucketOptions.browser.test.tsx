import type { CanvasDocumentContractV3, CanvasInteractionState } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import i18n from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/** The canvas document the form reads its selected layer from. */
const project = vi.hoisted(() => ({ document: null as unknown }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useActiveProjectSelector: (select: (value: { canvas: { document: unknown } }) => unknown) =>
    select({ canvas: { document: project.document } }),
}));

import { paintBucketForm } from './PaintBucketOptions';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const PaintBucketSettings = paintBucketForm.groups[0]!.body;

/** The engine's interaction capability over real stores, so writes and subscriptions behave as in the app. */
const engineOver = (stores: ReturnType<typeof createEngineStores>): ToolFormProps['engine'] =>
  ({
    interaction: {
      get: (key: keyof CanvasInteractionState) => stores[key].get(),
      set: (key: keyof CanvasInteractionState, value: never) => stores[key].set(value),
      subscribe: (key: keyof CanvasInteractionState, listener: () => void) => stores[key].subscribe(listener),
    },
  }) as unknown as ToolFormProps['engine'];

describe('paint bucket options', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  beforeAll(async () => {
    const translation = await fetch('/locales/en.json').then((response) => response.json());
    await i18n.use(initReactI18next).init({
      fallbackLng: 'en',
      initAsync: false,
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation } },
    });
  });

  afterEach(async () => {
    await act(() => root?.unmount());
    container?.remove();
    container = null;
    root = null;
  });

  /** Renders the form over real engine stores with `document` as the active project's canvas. */
  const renderForm = async (document: CanvasDocumentContractV3) => {
    project.document = document;
    const stores = createEngineStores('paintBucket');
    const engine = engineOver(stores);
    container = window.document.createElement('div');
    container.style.width = '320px';
    window.document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PaintBucketSettings engine={engine} />
        </ChakraProvider>
      );
    });
    return stores;
  };

  const slider = (name: string) => page.getByRole('slider', { exact: true, name });
  const rowSwitch = (name: string) => page.getByRole('checkbox', { exact: true, name });
  const stepUp = async (name: string) => {
    await act(async () => {
      (slider(name).element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight}');
    });
  };

  it('hides the edge and strength rows for a mask and binds Expand to the mask expansion', async () => {
    const stores = await renderForm(documentFrom([layerContract('Mask', 'inpaint_mask')], 'Mask'));

    await expect.element(page.getByText('Editing: Mask')).toBeVisible();
    await expect.element(page.getByRole('radiogroup', { name: 'Mode' })).toBeVisible();
    await expect.element(slider('Tolerance')).toHaveAttribute('aria-valuetext', '20%');
    await expect.element(rowSwitch('Contiguous')).toBeChecked();
    await expect.element(page.getByRole('radiogroup', { name: 'Sample' })).toBeVisible();
    await expect.element(slider('Close gaps')).toHaveAttribute('aria-valuetext', '0px');
    await expect.element(slider('Expand')).toHaveAttribute('aria-valuetext', '+2px');
    expect(container!.textContent).not.toContain('Anti-alias');
    expect(container!.textContent).not.toContain('Opacity');

    await stepUp('Expand');
    expect(stores.paintBucketOptions.get()).toMatchObject({ expandMask: 3, expandRaster: 0 });
  });

  it.each([
    ['raster', 'Paint'],
    ['control', 'Control'],
  ] as const)('shows every row for a %s layer and binds Expand to the raster expansion', async (type, name) => {
    const stores = await renderForm(documentFrom([layerContract(name, type)], name));

    await expect.element(page.getByText(`Editing: ${name}`)).toBeVisible();
    await expect.element(slider('Expand')).toHaveAttribute('aria-valuetext', '0px');
    await expect.element(rowSwitch('Anti-alias')).toBeChecked();
    await expect.element(slider('Opacity')).toHaveAttribute('aria-valuetext', '100%');

    await stepUp('Expand');
    expect(stores.paintBucketOptions.get()).toMatchObject({ expandMask: 2, expandRaster: 1 });
    await expect.element(slider('Expand')).toHaveAttribute('aria-valuetext', '+1px');
  });

  it('writes Mode and Sample, and disables Close gaps while Contiguous is off', async () => {
    const stores = await renderForm(documentFrom([layerContract('Paint', 'raster')], 'Paint'));

    await act(() => userEvent.click(page.getByText('Erase', { exact: true })));
    expect(stores.paintBucketOptions.get().mode).toBe('erase');
    await act(() => userEvent.click(page.getByText('Current layer', { exact: true })));
    expect(stores.paintBucketOptions.get().sample).toBe('layer');

    await expect.element(slider('Close gaps')).not.toHaveAttribute('aria-disabled', 'true');
    await act(() => userEvent.click(page.getByText('Contiguous', { exact: true })));
    expect(stores.paintBucketOptions.get().contiguous).toBe(false);
    await expect.element(slider('Close gaps')).toHaveAttribute('aria-disabled', 'true');
  });

  it('follows a tolerance scrubbed on the canvas, then the tolerance the release kept', async () => {
    const stores = await renderForm(documentFrom([layerContract('Paint', 'raster')], 'Paint'));
    await expect.element(slider('Tolerance')).toHaveAttribute('aria-valuetext', '20%');

    await act(() => stores.paintBucketScrub.set({ screenPoint: { x: 10, y: 10 }, tolerance: 47 }));
    await expect.element(slider('Tolerance')).toHaveAttribute('aria-valuetext', '47%');

    await act(() => {
      stores.paintBucketOptions.set({ ...stores.paintBucketOptions.get(), tolerance: 47 });
      stores.paintBucketScrub.set(null);
    });
    await expect.element(slider('Tolerance')).toHaveAttribute('aria-valuetext', '47%');
  });

  it.each([
    ['nothing', documentFrom([], null)],
    ['a group', documentFrom([groupContract('Folder', [layerContract('Paint', 'raster')])], 'Folder')],
    [
      'a text layer',
      documentFrom([layerContract('Title', 'raster', { source: { content: 'Hi', type: 'text' } as never })], 'Title'),
    ],
  ])('names a new layer as the fill target with %s selected, and asks for a layer to erase', async (_, document) => {
    const stores = await renderForm(document);
    await expect.element(page.getByText('Editing: New layer')).toBeVisible();

    await act(() => stores.paintBucketOptions.set({ ...stores.paintBucketOptions.get(), mode: 'erase' }));
    await expect.element(page.getByText('Select a layer to erase from.')).toBeVisible();
    expect(container!.textContent).not.toContain('Editing:');
    expect(container!.textContent).not.toContain('Defaults');
  });

  it('keeps naming the layer it erases from in Erase mode', async () => {
    const stores = await renderForm(documentFrom([layerContract('Paint', 'raster')], 'Paint'));
    await act(() => stores.paintBucketOptions.set({ ...stores.paintBucketOptions.get(), mode: 'erase' }));
    await expect.element(page.getByText('Editing: Paint')).toBeVisible();
    expect(container!.textContent).not.toContain('Select a layer to erase from.');
  });
});
