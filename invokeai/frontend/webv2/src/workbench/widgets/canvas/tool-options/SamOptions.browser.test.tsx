import type { CanvasOperationCapability, SamSessionSnapshot } from '@workbench/canvas-operations/api';

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

import { selectObjectOperationForm } from './SamOptions';

const SamSection = createOperationSection(selectObjectOperationForm);

const enCatalog: Record<string, unknown> = await fetch('/locales/en.json').then((response) => response.json());

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SESSION: SamSessionSnapshot = {
  applyPolygonRefinement: false,
  autoProcess: true,
  error: { code: 'queue', detail: 'AttributeError' },
  hasPreview: false,
  input: { prompt: 'cat', type: 'prompt' },
  invert: false,
  isolatedPreview: true,
  layerName: 'Portrait',
  layerType: 'raster',
  model: 'segment-anything-2-large',
  pointLabel: 'include',
  retrying: false,
  sourceRect: { height: 768, width: 1024, x: 0, y: 0 },
  status: 'error',
};

const engineWith = (operations: object): object => {
  const engine = {};
  attachCanvasOperations(engine, operations as never);
  return engine;
};

describe('Select Object retry focus', () => {
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    root = null;
    container = null;
  });

  type SetSession = (next: Partial<SamSessionSnapshot>) => void;
  type ProcessResult = Awaited<ReturnType<CanvasOperationCapability['processSelectObjectSession']>>;

  const startRetry = (set: SetSession): ProcessResult => {
    set({ error: null, retrying: true, status: 'processing-sam' });
    return 'published';
  };

  const renderFailedSession = async (process: (set: SetSession) => ProcessResult = startRetry) => {
    const i18n = createInstance();
    await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: enCatalog } } });
    let state = SESSION;
    const listeners = new Set<() => void>();
    const set: SetSession = (next) => {
      state = { ...state, ...next };
      listeners.forEach((listener) => listener());
    };
    const operations = {
      getSamSessionState: () => state,
      processSelectObjectSession: vi.fn(() => Promise.resolve(process(set))),
      subscribeSamSession: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      updateSelectObjectSession: vi.fn(() => 'updated' as const),
    };
    const engine = engineWith(operations);
    container = document.createElement('div');
    container.style.width = '450px';
    document.body.append(container);
    root = createRoot(container);
    act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <SamSection engine={engine as never} />
          </I18nextProvider>
        </ChakraProvider>
      )
    );
    const findObject = page.getByRole('button', { name: 'Find object' });
    await expect.element(findObject).toBeEnabled();
    return {
      fail: () => act(() => set({ error: { code: 'queue' }, retrying: false, status: 'error' })),
      findObject,
      operations,
      set,
      succeed: async () => {
        await expect.element(findObject).toBeDisabled();
        act(() => set({ hasPreview: true, retrying: false, status: 'ready' }));
        expect(findObject.query()).toBeNull();
      },
    };
  };

  it('moves focus to Apply when a keyboard retry succeeds and Find object leaves', async () => {
    const { findObject, operations, succeed } = await renderFailedSession();
    findObject.element().focus();
    await userEvent.keyboard('{Enter}');
    expect(operations.processSelectObjectSession).toHaveBeenCalledOnce();

    await succeed();

    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toHaveFocus();
  });

  it('does not move focus when Find object leaves later, after the keyboard retry failed', async () => {
    const { fail, findObject, set } = await renderFailedSession();
    findObject.element().focus();
    await userEvent.keyboard('{Enter}');
    await expect.element(findObject).toBeDisabled();
    fail();
    await expect.element(findObject).toBeEnabled();
    (document.activeElement as HTMLElement | null)?.blur();

    act(() => set({ error: null, hasPreview: true, status: 'ready' }));
    expect(findObject.query()).toBeNull();
    await act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve);
        })
    );

    expect(document.activeElement).toBe(document.body);
  });

  it('moves focus to Apply when a keyboard retry succeeds before it renders as running', async () => {
    const { findObject } = await renderFailedSession((set) => {
      set({ error: null, hasPreview: true, status: 'ready' });
      return 'published';
    });
    findObject.element().focus();
    await userEvent.keyboard('{Enter}');
    expect(findObject.query()).toBeNull();

    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toHaveFocus();
  });

  it.each([
    { name: 'is refused', process: (): ProcessResult => 'deduped' },
    {
      name: 'fails before it renders as running',
      process: (set: SetSession): ProcessResult => {
        startRetry(set);
        set({ error: { code: 'queue' }, retrying: false, status: 'error' });
        return 'error';
      },
    },
  ])('does not move focus when Find object leaves later, after a keyboard retry that $name', async ({ process }) => {
    const { findObject, operations, set } = await renderFailedSession(process);
    findObject.element().focus();
    await userEvent.keyboard('{Enter}');
    expect(operations.processSelectObjectSession).toHaveBeenCalledOnce();
    await expect.element(findObject).toBeEnabled();
    (document.activeElement as HTMLElement | null)?.blur();

    act(() => set({ error: null, hasPreview: true, status: 'ready' }));
    expect(findObject.query()).toBeNull();
    await act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve);
        })
    );

    expect(document.activeElement).toBe(document.body);
  });

  it('leaves focus where the user moved it during the retry', async () => {
    const { findObject, succeed } = await renderFailedSession();
    findObject.element().focus();
    await userEvent.keyboard('{Enter}');
    const prompt = page.getByRole('textbox', { name: 'Prompt' });
    prompt.element().focus();

    await succeed();

    await expect.element(prompt).toHaveFocus();
  });

  it('keeps focus in the prompt for a retry started by Enter there', async () => {
    const { operations, succeed } = await renderFailedSession();
    const prompt = page.getByRole('textbox', { name: 'Prompt' });
    await userEvent.click(prompt);
    await userEvent.keyboard('{Enter}');
    expect(operations.processSelectObjectSession).toHaveBeenCalledOnce();

    await succeed();

    await expect.element(prompt).toHaveFocus();
  });
});
