import type { FilterOperationSessionState } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { filterOperationForm } from './FilterOptions';

const enCatalog: Record<string, unknown> = await fetch('/locales/en.json').then((response) => response.json());

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createI18n = async () => {
  const i18n = createInstance();
  await i18n.use(initReactI18next).init({
    fallbackLng: 'en',
    lng: 'en',
    resources: {
      // A partial catalog: only the over-budget refusal is translated.
      de: { translation: { widgets: { layers: { rasterFilter: { overBudget: 'Zu groß zum Rückgängigmachen.' } } } } },
      en: { translation: enCatalog },
    },
  });
  return i18n;
};

const sessionStore = (initial: FilterOperationSessionState) => {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    getFilterSessionState: () => state,
    set: (next: Partial<FilterOperationSessionState>) => {
      state = { ...state, ...next };
      listeners.forEach((listener) => listener());
    },
    subscribeFilterSession: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const engineWith = (operations: ReturnType<typeof sessionStore>): object => {
  const engine = {};
  attachCanvasOperations(engine, operations as never);
  return engine;
};

const SESSION: FilterOperationSessionState = {
  autoProcess: true,
  draft: { settings: {}, type: 'canny_edge_detection' },
  error: null,
  initialFilter: null,
  layerId: 'layer-1',
  layerName: 'Portrait',
  layerType: 'raster',
  preview: null,
  status: 'ready',
};

describe('filter footer errors', () => {
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    root = null;
    container = null;
  });

  const renderFooter = async (initial: Partial<FilterOperationSessionState>) => {
    const i18n = await createI18n();
    const store = sessionStore({ ...SESSION, ...initial });
    const engine = engineWith(store);
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    const Footer = filterOperationForm.footer;
    act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <Footer engine={engine as never} isExternalInteractionLocked={false} />
          </I18nextProvider>
        </ChakraProvider>
      )
    );
    return { i18n, store };
  };

  it('translates a refusal and updates an already visible error when the language changes', async () => {
    const { i18n } = await renderFooter({ error: { code: 'over-budget' }, status: 'error' });
    await expect
      .element(page.getByRole('alert'))
      .toHaveTextContent('The filtered layer is too large to undo, so it was not applied.');

    await act(() => i18n.changeLanguage('de'));

    await expect.element(page.getByRole('alert')).toHaveTextContent('Zu groß zum Rückgängigmachen.');
  });

  it('falls back to English for a refusal the language does not translate', async () => {
    const { i18n } = await renderFooter({ error: { code: 'stale' }, status: 'error' });
    await act(() => i18n.changeLanguage('de'));

    await expect
      .element(page.getByRole('alert'))
      .toHaveTextContent('The layer changed while filtering. Preview it again.');
  });

  it('summarizes an unexpected failure and keeps its message as technical detail', async () => {
    await renderFooter({ error: { code: 'apply-failed', detail: 'cache failed' }, status: 'error' });

    await expect.element(page.getByRole('alert')).toHaveTextContent('The filter could not be applied.');
    const details = page.getByRole('button', { name: 'Technical details' });
    await userEvent.hover(details);
    await expect.element(page.getByText('cache failed')).toBeVisible();
  });

  it('clears the error once a retry starts', async () => {
    const { store } = await renderFooter({ error: { code: 'busy' }, status: 'error' });
    await expect.element(page.getByRole('alert')).toBeVisible();

    act(() => store.set({ error: null, status: 'committing' }));

    await expect.element(page.getByRole('alert')).not.toBeInTheDocument();
    await expect.element(page.getByRole('status')).toHaveTextContent('Applying filter…');
  });
});
