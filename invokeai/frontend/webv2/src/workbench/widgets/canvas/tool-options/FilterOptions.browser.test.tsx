import type { FilterOperationSessionState } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createOperationSection } from '@workbench/widgets/layers/panes/OperationSection';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { filterOperationForm } from './FilterOptions';

const FilterSection = createOperationSection(filterOperationForm);

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
    processFilterOperation: vi.fn(() => Promise.resolve<'completed' | 'stale'>('completed')),
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

describe('filter operation errors', () => {
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    root = null;
    container = null;
  });

  const renderSection = async (initial: Partial<FilterOperationSessionState>) => {
    const i18n = await createI18n();
    const store = sessionStore({ ...SESSION, ...initial });
    const engine = engineWith(store);
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <FilterSection engine={engine as never} />
          </I18nextProvider>
        </ChakraProvider>
      )
    );
    return { i18n, store };
  };

  it('translates a refusal and updates an already visible error when the language changes', async () => {
    const { i18n } = await renderSection({ error: { code: 'over-budget' }, status: 'error' });
    await expect
      .element(page.getByRole('alert'))
      .toHaveTextContent('The filtered layer is too large to undo, so it was not applied.');

    await act(() => i18n.changeLanguage('de'));

    await expect.element(page.getByRole('alert')).toHaveTextContent('Zu groß zum Rückgängigmachen.');
  });

  it('falls back to English for a refusal the language does not translate', async () => {
    const { i18n } = await renderSection({ error: { code: 'stale' }, status: 'error' });
    await act(() => i18n.changeLanguage('de'));

    await expect
      .element(page.getByRole('alert'))
      .toHaveTextContent('The layer changed while filtering. Preview it again.');
  });

  it('summarizes an unexpected failure and keeps its message as technical detail', async () => {
    await renderSection({ error: { code: 'apply-failed', detail: 'cache failed' }, status: 'error' });

    await expect.element(page.getByRole('alert')).toHaveTextContent('The filter could not be applied.');
    const details = page.getByRole('button', { name: 'Technical details' });
    await userEvent.hover(details);
    await expect.element(page.getByText('cache failed')).toBeVisible();
  });

  it('offers Process to retry a failed run with auto-update on, until that retry has finished', async () => {
    const { store } = await renderSection({});
    const process = page.getByRole('button', { exact: true, name: 'Process' });
    expect(process.query()).toBeNull();

    act(() => store.set({ error: { code: 'process-failed', detail: 'graph exploded' }, status: 'error' }));
    await expect.element(process).toBeEnabled();
    let finish!: () => void;
    store.processFilterOperation.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = () => resolve('completed');
        })
    );
    await userEvent.click(process);
    expect(store.processFilterOperation).toHaveBeenCalledOnce();

    // The retry clears the error as it starts; the button stays, busy, for the run it started.
    act(() => store.set({ error: null, status: 'processing' }));
    await expect.element(process).toBeDisabled();
    act(() => store.set({ status: 'ready' }));
    await act(() => finish());
    expect(process.query()).toBeNull();
  });

  it('leaves a failed apply to Apply: no Process with auto-update on', async () => {
    await renderSection({
      error: { code: 'apply-failed', detail: 'cache failed' },
      preview: {
        guard: {} as never,
        height: 10,
        imageName: 'filtered',
        origin: { x: 0, y: 0 },
        rect: { height: 10, width: 10, x: 0, y: 0 },
        width: 10,
      },
      status: 'error',
    });

    await expect.element(page.getByRole('alert')).toHaveTextContent('The filter could not be applied.');
    expect(page.getByRole('button', { exact: true, name: 'Process' }).query()).toBeNull();
    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toBeEnabled();
  });

  it('clears the error once a retry starts', async () => {
    const { store } = await renderSection({ error: { code: 'busy' }, status: 'error' });
    await expect.element(page.getByRole('alert')).toBeVisible();

    act(() => store.set({ error: null, status: 'committing' }));

    await expect.element(page.getByRole('alert')).not.toBeInTheDocument();
    await expect.element(page.getByRole('status')).toHaveTextContent('Applying filter…');
  });
});
