/* eslint-disable react-perf/jsx-no-new-object-as-prop -- test injects a fake workflow UI adapter */
import type { InvocationTemplate } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
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

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, status: 'loaded', templates: TEMPLATES }),
}));

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

  it('does not mount the virtualized result content while closed', async () => {
    const errorSpy = vi.spyOn(console, 'error');

    try {
      await act(() => {
        root.render(
          <StrictMode>
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
          </StrictMode>
        );
      });

      expect(host.querySelector('[aria-label="Node search results"]')).toBeNull();
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
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const renderOpen = async ({
    groupByCategory = true,
    isOpen = true,
    registerModalHotkeyLayer = (): (() => void) => () => {},
  } = {}) => {
    const onAddNode = vi.fn();
    const adapter = {
      getProjectGraph: () => ({ edges: [], nodes: [] }),
      preferences: {
        getSnapshot: () => ({ workflowGroupNodesByCategory: groupByCategory }),
        subscribe: () => () => {},
      },
      registerModalHotkeyLayer,
    } as unknown as WorkflowUiAdapter;

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <AddNodeDialog
              connectionFilter={null}
              isOpen={isOpen}
              onAddConnector={vi.fn()}
              onAddCurrentImage={vi.fn()}
              onAddNode={onAddNode}
              onAddNote={vi.fn()}
              onOpenChange={vi.fn()}
            />
          </WorkflowUiProvider>
        </ChakraProvider>
      );
    });

    return { onAddNode, search: document.querySelector<HTMLInputElement>('input[aria-label="Search for nodes"]')! };
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
    expect(document.querySelector('[aria-label="Expand all categories"]')).toBeNull();
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
    let isDialogShownAtRelease: boolean | null = null;
    const registerModalHotkeyLayer = vi.fn(
      () => () => (isDialogShownAtRelease = document.querySelector('[role="dialog"]') !== null)
    );
    await renderOpen({ registerModalHotkeyLayer });
    expect(registerModalHotkeyLayer).toHaveBeenCalledExactlyOnceWith('workflow-add-node');

    await renderOpen({ isOpen: false, registerModalHotkeyLayer });
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

    expect(isDialogShownAtRelease).toBe(true);
    expect(registerModalHotkeyLayer).toHaveBeenCalledOnce();
  });

  it('adds the best match on Enter instead of toggling its category', async () => {
    const { onAddNode, search } = await renderOpen();

    await act(() => userEvent.type(search, 'integer{Enter}'));

    expect(onAddNode).toHaveBeenCalledExactlyOnceWith(TEMPLATES.integer);
  });
});
