import type { CanvasInteractionState } from '@workbench/canvas-engine/api';
import type { ToolFooterProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { transformForm } from './geometryForm';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

/** The engine's interaction store as the footer reads it: a value per key and a listener set per key. */
const createEngine = () => {
  const state: Partial<CanvasInteractionState> = {
    hasFloatingSelection: false,
    pendingTransformLayerId: null,
    transformSession: null,
  };
  const listeners = new Map<string, Set<() => void>>();
  const engine = {
    interaction: {
      get: (key: keyof CanvasInteractionState) => state[key],
      subscribe: (key: string, listener: () => void) => {
        const keyListeners = listeners.get(key) ?? new Set();
        keyListeners.add(listener);
        listeners.set(key, keyListeners);
        return () => keyListeners.delete(listener);
      },
    },
    layers: { applyTransform: vi.fn(), cancelTransform: vi.fn() },
  };
  const setPending = (layerId: string | null) =>
    act(() => {
      state.pendingTransformLayerId = layerId;
      listeners.get('pendingTransformLayerId')?.forEach((listener) => listener());
    });
  return { engine: engine as unknown as ToolFooterProps['engine'], setPending };
};

describe('Transform footer', () => {
  it('says while a transform waits for its stroke to save, in a polite live region, and clears after', async () => {
    const { engine, setPending } = createEngine();
    const Footer = transformForm.footer!;
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Footer engine={engine} isExternalInteractionLocked={false} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
    const status = page.getByRole('status');
    await expect.element(status).toBeInTheDocument();
    await expect.element(status).toHaveTextContent('');

    await setPending('layer-1');
    await expect.element(status).toHaveTextContent('Waiting for the stroke to save…');
    await expect.element(page.getByRole('button', { name: 'Apply' })).toBeDisabled();

    await setPending(null);
    await expect.element(status).toHaveTextContent('');
  });
});
