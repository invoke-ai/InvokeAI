/* eslint-disable react-perf/jsx-no-new-object-as-prop -- test injects a fake workflow UI adapter */
import type { InvocationTemplate } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { AddNodeConnectionFilter } from '@features/workflow/ui/workflowUiStore';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { isModalPresent } from '@platform/ui/modalPresence';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { AddNodeDialog } from './AddNodeDialog';

const template = (type: string, title: string, category: string, tags: string[] = []): InvocationTemplate => ({
  category,
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputType: `${type}_output`,
  outputs: {},
  tags,
  title,
  type,
  useCache: true,
  version: '1.0.0',
});

const TEMPLATES = Object.fromEntries(
  [
    template('add', 'Add Integers', 'math', ['math', 'integer']),
    template('integer_collection', 'Integer Collection', 'primitives'),
    template('integer', 'Integer', 'primitives'),
    template('img_resize', 'Resize Image', 'image'),
    template('i2l', 'Image to Latents', 'latents'),
    template('l2i', 'Latents to Pixels', 'latents', ['image']),
  ].map((entry) => [entry.type, entry])
);

/** A stand-in for the template store: a test may fail the load, and a retry asks it to load again. */
const templatesStore = vi.hoisted(() => {
  type Snapshot = { error: string | null; status: 'error' | 'loaded' | 'loading'; templates: object };
  const listeners = new Set<() => void>();
  const store = {
    ensureLoaded: vi.fn(() => store.set({ error: null, status: 'loading', templates: {} })),
    getSnapshot: (): Snapshot => store.snapshot,
    set: (snapshot: Snapshot) => {
      store.snapshot = snapshot;
      listeners.forEach((listener) => listener());
    },
    snapshot: { error: null, status: 'loaded', templates: {} } as Snapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  return store;
});

vi.mock('@features/workflow/react', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');

  return {
    ...(await importOriginal<Record<string, unknown>>()),
    ensureInvocationTemplatesLoaded: templatesStore.ensureLoaded,
    useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
      selector(useSyncExternalStore(templatesStore.subscribe, templatesStore.getSnapshot)),
  };
});

const i18n = createInstance();
await i18n.init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('AddNodeDialog lifecycle', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  beforeEach(() => {
    templatesStore.ensureLoaded.mockClear();
    templatesStore.snapshot = { error: null, status: 'loaded', templates: TEMPLATES };
  });

  it('does not mount the virtualized result content while closed', async () => {
    const errorSpy = vi.spyOn(console, 'error');

    try {
      await act(() => {
        root.render(
          <StrictMode>
            <I18nextProvider i18n={i18n}>
              <ChakraProvider value={system}>
                <AddNodeDialog
                  connectionFilter={null}
                  isOpen={false}
                  onAddConnector={vi.fn()}
                  onAddCurrentImage={vi.fn()}
                  onAddNode={vi.fn()}
                  onAddNote={vi.fn()}
                  onOpenChange={vi.fn()}
                />
              </ChakraProvider>
            </I18nextProvider>
          </StrictMode>
        );
      });

      expect(document.querySelector('[role="tree"]')).toBeNull();
      expect(errorSpy.mock.calls.flat().join(' ')).not.toContain('getSnapshot');
    } finally {
      errorSpy.mockRestore();
    }
  });
});

describe('AddNodeDialog search', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    templatesStore.ensureLoaded.mockClear();
    templatesStore.snapshot = { error: null, status: 'loaded', templates: TEMPLATES };
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const renderOpen = async ({
    connectionFilter = null,
    groupByCategory = true,
    isOpen = true,
  }: { connectionFilter?: AddNodeConnectionFilter | null; groupByCategory?: boolean; isOpen?: boolean } = {}) => {
    const onAddNode = vi.fn();
    const adapter = {
      getProjectGraph: () => ({ edges: [], nodes: [] }),
      preferences: {
        getSnapshot: () => ({ workflowGroupNodesByCategory: groupByCategory }),
        subscribe: () => () => {},
      },
    } as unknown as WorkflowUiAdapter;

    await act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <AddNodeDialog
                connectionFilter={connectionFilter}
                isOpen={isOpen}
                onAddConnector={vi.fn()}
                onAddCurrentImage={vi.fn()}
                onAddNode={onAddNode}
                onAddNote={vi.fn()}
                onOpenChange={vi.fn()}
              />
            </WorkflowUiProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });

    return { onAddNode, search: document.querySelector<HTMLInputElement>('input[role="combobox"]')! };
  };

  const listedRows = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[role="treeitem"]')).map((row) =>
      row.getAttribute('aria-expanded') === null
        ? (row.querySelector('[title]')?.getAttribute('title') ?? row.textContent?.replace(/invokeai$/, '') ?? '')
        : `# ${row.textContent?.replace(/\d+$/, '')}`
    );

  it('lists the closest names first, with the category holding the best match leading', async () => {
    const { search } = await renderOpen();

    await act(() => userEvent.type(search, 'integer'));

    expect(listedRows()).toEqual(['# Primitives', 'Integer', 'Integer Collection', '# Math', 'Add Integers']);
  });

  it('lists one ranked list without category headers when grouping is off', async () => {
    const { search } = await renderOpen({ groupByCategory: false });

    await act(() => userEvent.type(search, 'image'));

    // A name prefix leads, the utility row leads its tie, and a tag-only match ranks below every name match.
    expect(listedRows()).toEqual(['Image to Latents', 'Current Image', 'Resize Image', 'Latents to Pixels']);
    expect(
      document.querySelector(`[aria-label="${i18n.t('widgets.workflow.addNodeDialog.expandAllCategories')}"]`)
    ).toBeNull();
  });

  it('animates out on close, then starts the next open with an empty search', async () => {
    const { search } = await renderOpen();
    await act(() => userEvent.type(search, 'integer'));
    const dialog = document.querySelector('[role="dialog"]')!;

    const frames = closingFrames(await recordDialogExit(dialog, () => renderOpen({ isOpen: false })));
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

    expect(frames).not.toHaveLength(0);
    expect((await renderOpen()).search.value).toBe('');
  });

  it('gives hotkeys back to the workbench as it closes, not after it animates out', async () => {
    await renderOpen();
    expect(isModalPresent()).toBe(true);
    const dialog = document.querySelector('[role="dialog"]')!;
    // A close during the opening animation unmounts at once; settle it so the close animates out.
    await expect.poll(() => dialog.getAnimations().length).toBe(0);

    await renderOpen({ isOpen: false });
    await expect.poll(() => isModalPresent()).toBe(false);

    expect(dialog.isConnected).toBe(true);
    expect(dialog.getAttribute('data-state')).toBe('closed');
  });

  it('adds the best match on Enter instead of toggling its category', async () => {
    const { onAddNode, search } = await renderOpen();

    await act(() => userEvent.type(search, 'integer{Enter}'));

    expect(onAddNode).toHaveBeenCalledExactlyOnceWith(TEMPLATES.integer);
  });

  it('leaves arrows and Enter to an IME while it composes', async () => {
    const { onAddNode, search } = await renderOpen();
    await act(() => userEvent.type(search, 'integer'));
    const activeRow = () => document.getElementById(search.getAttribute('aria-activedescendant')!)?.textContent;
    const press = (key: string, init: KeyboardEventInit) =>
      act(() => {
        search.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key, ...init }));
      });
    const bestMatch = activeRow();

    await press('ArrowDown', { isComposing: true });
    await press('Enter', { isComposing: true });
    // Safari's confirming keydown arrives after compositionend, marked only by keyCode 229.
    await press('Enter', { keyCode: 229 });

    expect(activeRow()).toBe(bestMatch);
    expect(onAddNode).not.toHaveBeenCalled();
  });

  /** Status messages render beside the result tree, inside the dialog's scroll area. */
  const resultsArea = () => document.querySelector('[role="tree"]')!.parentElement!;

  it('names its search and results, and says what to try when nothing matches', async () => {
    const { search } = await renderOpen();
    const results = document.querySelector('[role="tree"]')!;

    expect(search.getAttribute('aria-label')).toBe('Search for nodes');
    expect(search.placeholder).toBe('Search for nodes…');
    expect(search.getAttribute('aria-controls')).toBe(results.id);
    expect(results.getAttribute('aria-label')).toBe('Node search results');

    await act(() => userEvent.type(search, 'zzzz'));

    expect(results.children).toHaveLength(0);
    expect(resultsArea().textContent).toBe('No nodes match your search. Try another name, node type, tag or category.');
    expect(search.hasAttribute('aria-activedescendant')).toBe(false);
  });

  it('names the type a pending connection needs while searching for a compatible node', async () => {
    const { search } = await renderOpen({
      connectionFilter: {
        kind: 'source',
        sourceHandle: 'value',
        sourceNodeId: 'source',
        sourceType: { batch: false, cardinality: 'COLLECTION', name: 'IntegerField' },
      },
    });

    // The field type reads as node handles name it, not as its schema name.
    expect(search.placeholder).toBe('Search nodes compatible with Integer Collection…');

    await act(() => userEvent.type(search, 'zzzz'));

    expect(resultsArea().textContent).toBe(
      'No nodes compatible with Integer Collection match your search. Try another name, node type, tag or category.'
    );
  });

  it('retries a failed node list load, busy and focused until the nodes arrive', async () => {
    templatesStore.snapshot = { error: 'Service unavailable', status: 'error', templates: {} };

    const { search } = await renderOpen();
    const alert = document.querySelector<HTMLElement>('[role="alert"]')!;
    const retry = [...alert.querySelectorAll('button')].find((button) => button.textContent === 'Retry')!;

    expect(alert.textContent).toBe(
      'Could not load the node list. Check the connection to the server and try again.Service unavailableRetry'
    );

    // Wait for modal setup to isolate the background before moving focus to Retry.
    await expect.poll(() => host.getAttribute('aria-hidden')).toBe('true');
    retry.focus();
    await act(() => userEvent.keyboard('{Enter}'));

    expect(templatesStore.ensureLoaded).toHaveBeenCalledOnce();
    expect(retry.getAttribute('aria-busy')).toBe('true');
    expect(retry.getAttribute('aria-disabled')).toBe('true');
    expect(document.activeElement).toBe(retry);

    // Playwright will not click an aria-disabled control; a direct click checks the busy Retry ignores it.
    await act(() => retry.click());
    expect(templatesStore.ensureLoaded).toHaveBeenCalledOnce();

    await act(async () => {
      templatesStore.set({ error: null, status: 'loaded', templates: TEMPLATES });
      await Promise.resolve();
    });

    expect(document.querySelector('[role="alert"]')).toBeNull();
    expect(document.querySelectorAll('[role="tree"] [role="treeitem"]').length).toBeGreaterThan(0);
    expect(document.activeElement).toBe(search);
  });

  it('leaves out the server detail when the failure has none', async () => {
    templatesStore.snapshot = { error: null, status: 'error', templates: {} };

    await renderOpen();

    expect(document.querySelector('[role="alert"]')?.textContent).toBe(
      'Could not load the node list. Check the connection to the server and try again.Retry'
    );
  });
});
