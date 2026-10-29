import type { ProjectGraphState } from '@features/workflow/contracts';
import type * as workflowReact from '@features/workflow/react';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/react';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { LinearFormView } from './LinearFormView';

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof workflowReact>()),
  useInvocationTemplatesSelector: (select: (snapshot: unknown) => unknown) =>
    select({ error: null, status: 'loaded', templates: {} }),
}));

const i18n = createInstance();
await i18n.init({
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((r) => r.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const graph: ProjectGraphState = {
  ...createProjectGraph('accordion-test'),
  form: {
    rootElementId: 'root',
    elements: {
      root: {
        id: 'root',
        type: 'container',
        data: {
          layout: 'column',
          children: ['intro', 'input', 'missing', 'divider', 'between', 'output', 'result', 'boundary', 'tail', 'end'],
        },
      },
      intro: { id: 'intro', type: 'text', parentId: 'root', data: { content: 'Always visible' } },
      input: { id: 'input', type: 'heading', parentId: 'root', data: { content: 'Input section' } },
      missing: {
        id: 'missing',
        type: 'node-field',
        parentId: 'root',
        data: {
          fieldIdentifier: { nodeId: 'missing-node', fieldName: 'value' },
          showDescription: false,
          showShuffle: false,
        },
      },
      divider: { id: 'divider', type: 'divider', parentId: 'root' },
      between: { id: 'between', type: 'text', parentId: 'root', data: { content: 'Between sections' } },
      output: { id: 'output', type: 'heading', parentId: 'root', data: { content: 'Output section' } },
      result: { id: 'result', type: 'text', parentId: 'root', data: { content: 'A retained value' } },
      boundary: { id: 'boundary', type: 'divider', parentId: 'root' },
      tail: { id: 'tail', type: 'heading', parentId: 'root', data: { content: 'Final section' } },
      end: { id: 'end', type: 'text', parentId: 'root', data: { content: 'Final value' } },
    },
  },
};

let host: HTMLDivElement;
let root: Root;
let snapshot: { id: string; projectGraph: ProjectGraphState; workflowValues: Record<string, unknown> };
const listeners = new Set<() => void>();
const project = {
  getSnapshot: () => snapshot,
  subscribe: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};
const widgets = {
  open: vi.fn(),
  patchValues: vi.fn((_id: string, values: Record<string, unknown>) => {
    snapshot = { ...snapshot, workflowValues: { ...snapshot.workflowValues, ...values } };
    listeners.forEach((listener) => listener());
  }),
};
const adapter = { commands: { editGraph: vi.fn() }, project, widgets } as unknown as WorkflowUiAdapter;

const renderForm = async (sectionGapPx?: number) => {
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WorkflowUiProvider adapter={adapter}>
            <LinearFormView projectGraph={graph} sectionGapPx={sectionGapPx} />
          </WorkflowUiProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  snapshot = { id: 'project-1', projectGraph: graph, workflowValues: {} };
  listeners.clear();
  widgets.patchValues.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  listeners.clear();
});

describe('native workflow-form accordion', () => {
  it('collapses only fields under the clicked heading, preserving errors and later siblings', async () => {
    await renderForm();
    const input = page.getByRole('button', { name: /^Input section/ });
    await expect.element(input).toHaveAttribute('aria-expanded', 'true');
    await act(() => userEvent.click(input));
    await expect.element(input).toHaveAttribute('aria-expanded', 'false');
    await expect.element(page.getByText('Needs attention')).toBeVisible();
    await expect.element(page.getByText('Always visible')).toBeVisible();
    await expect.element(page.getByText('Between sections')).toBeVisible();
    await expect.element(page.getByText('A retained value')).toBeVisible();
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [graph.id]: { input: false } });
  });

  it('restores the same heading state after remount and reopens from the keyboard', async () => {
    await renderForm();
    const output = page.getByRole('button', { name: 'Output section' });
    await act(() => userEvent.click(output));
    await expect.element(page.getByText('A retained value')).not.toBeVisible();

    await act(() => root.unmount());
    root = createRoot(host);
    await renderForm();
    const restored = page.getByRole('button', { name: 'Output section' });
    await expect.element(restored).toHaveAttribute('aria-expanded', 'false');
    restored.element().focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.element(page.getByText('A retained value')).toBeVisible();
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [graph.id]: { output: true } });
  });

  it('uses the chosen section gap, hides only inter-section dividers, and leaves collapsed headers unfilled', async () => {
    await renderForm(8);
    const stack = host.querySelector<HTMLElement>('.chakra-stack')!;
    expect(getComputedStyle(stack).gap).toBe('8px');
    expect(host.querySelectorAll('[role="separator"]')).toHaveLength(1);

    const input = page.getByRole('button', { name: /^Input section/ });
    await act(() => userEvent.click(input));
    await expect.element(input).toHaveAttribute('aria-expanded', 'false');
    const section = input.element().closest<HTMLElement>('[data-scope="collapsible"][data-part="root"]')!;
    expect(getComputedStyle(section).backgroundColor).toBe('rgba(0, 0, 0, 0)');
    expect(getComputedStyle(stack).gap).toBe('8px');
  });
});
