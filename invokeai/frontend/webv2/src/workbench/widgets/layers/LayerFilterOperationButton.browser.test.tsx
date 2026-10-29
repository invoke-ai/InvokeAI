import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it } from 'vitest';
import { page } from 'vitest/browser';

import { LayerFilterOperationButton, type LayerFilterOperationEngine } from './LayerFilterOperationButton';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: { 'widgets.layers.control.filter': 'Filter' } } },
});

/** An engine whose layer gains content later, announced through one of its interaction signals. */
const createChangingEngine = () => {
  let hasContent = false;
  const listeners = new Map<string, Set<() => void>>();
  const engine = {
    exports: { hasExportableLayerContent: () => hasContent },
    interaction: {
      subscribe: (key: string, listener: () => void) => {
        const set = listeners.get(key) ?? new Set();
        set.add(listener);
        listeners.set(key, set);
        return () => set.delete(listener);
      },
    },
    projectId: 'project',
  } as unknown as LayerFilterOperationEngine;
  const gainContent = (signal: 'layerPixelEpoch' | 'documentEpoch'): void => {
    hasContent = true;
    listeners.get(signal)?.forEach((listener) => listener());
  };
  return { engine, gainContent };
};

const noop = (): void => {};

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

it.each([
  ['publishes pixels (a fill)', 'layerPixelEpoch'],
  ['gains content through a document change', 'documentEpoch'],
] as const)('enables Filter as soon as the layer %s, without reselecting it', async (_, signal) => {
  const { engine, gainContent } = createChangingEngine();
  const layer = createEmptyPaintLayer('Raster', 'raster');
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <LayerFilterOperationButton engine={engine} layer={layer} onOperationStarted={noop} operations={null} />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  await expect.element(page.getByRole('button', { name: 'Filter' })).toBeDisabled();

  act(() => gainContent(signal));

  await expect.element(page.getByRole('button', { name: 'Filter' })).toBeEnabled();
});
