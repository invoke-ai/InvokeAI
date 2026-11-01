import type { InvocationTemplate, ProjectGraphState } from '@features/workflow/contracts';
import type * as workflowReact from '@features/workflow/react';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerateCollapsibleSection } from '@features/generation/ui/shared/GenerateCollapsibleSection';
import { WorkflowUiProvider } from '@features/workflow/react';
import { buildInvocationNode, createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { LinearFormView } from './LinearFormView';

const numericInput = (name: string, title: string, value: number): InvocationTemplate['inputs'][string] => ({
  default: value,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'direct',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title,
  type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
});
const numericTemplate: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: { first: numericInput('first', 'First value', 11), second: numericInput('second', 'Second value', 22) },
  nodePack: 'invokeai',
  outputType: 'test',
  outputs: {},
  tags: [],
  title: 'Numeric fields',
  type: 'numeric-test',
  useCache: true,
  version: '1.0.0',
};

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof workflowReact>()),
  useInvocationTemplatesSelector: (select: (snapshot: unknown) => unknown) =>
    select({ error: null, status: 'loaded', templates: { [numericTemplate.type]: numericTemplate } }),
}));

vi.mock('@features/generation/ui/GenerationUiContext', () => ({
  useGenerationUi: () => ({ sectionPreferences: { sectionsOpen: {}, setSectionOpen: vi.fn() } }),
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

const layoutGraph = (layout: 'row' | 'column', atRoot = false): ProjectGraphState => {
  const node = { ...buildInvocationNode(numericTemplate, { x: 0, y: 0 }), id: 'numeric-node' };
  const containerId = atRoot ? 'root' : 'group';
  return {
    ...createProjectGraph(`layout-${layout}-${atRoot}`),
    nodes: [node],
    form: {
      rootElementId: 'root',
      elements: {
        root: {
          id: 'root',
          type: 'container',
          data: {
            layout: atRoot ? layout : 'column',
            children: atRoot ? ['heading', 'first', 'second'] : ['before', 'group', 'after'],
          },
        },
        ...(!atRoot
          ? {
              before: { id: 'before', type: 'text' as const, parentId: 'root', data: { content: 'Before row' } },
              group: {
                id: 'group',
                type: 'container' as const,
                parentId: 'root',
                data: { layout, children: ['heading', 'first', 'second'] },
              },
              after: { id: 'after', type: 'text' as const, parentId: 'root', data: { content: 'After row' } },
            }
          : {}),
        heading: { id: 'heading', type: 'heading', parentId: containerId, data: { content: 'Numeric section' } },
        first: {
          id: 'first',
          type: 'node-field',
          parentId: containerId,
          data: {
            fieldIdentifier: { nodeId: node.id, fieldName: 'first' },
            showDescription: false,
            showShuffle: false,
          },
        },
        second: {
          id: 'second',
          type: 'node-field',
          parentId: containerId,
          data: {
            fieldIdentifier: { nodeId: node.id, fieldName: 'second' },
            showDescription: false,
            showShuffle: false,
          },
        },
      },
    },
  };
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

const renderForm = async (sectionGapPx?: number, projectGraph = graph, withNativeReference = false) => {
  snapshot = { ...snapshot, projectGraph };
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WorkflowUiProvider adapter={adapter}>
            <LinearFormView projectGraph={projectGraph} sectionGapPx={sectionGapPx} />
          </WorkflowUiProvider>
          {withNativeReference ? (
            <div data-testid="native-reference">
              <GenerateCollapsibleSection label="Native section" defaultOpen>
                Native content
              </GenerateCollapsibleSection>
              <GenerateCollapsibleSection label="Native second section" defaultOpen>
                Native second content
              </GenerateCollapsibleSection>
            </div>
          ) : null}
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

const visibleBoundaries = () =>
  [...host.querySelectorAll<HTMLElement>('*')].filter((element) => {
    const style = getComputedStyle(element, '::before');
    return (
      element.getBoundingClientRect().width > 0 &&
      style.content === '""' &&
      style.borderTopWidth === '1px' &&
      style.opacity === '1' &&
      style.pointerEvents === 'none'
    );
  });

beforeEach(() => {
  snapshot = { id: 'project-1', projectGraph: graph, workflowValues: {} };
  listeners.clear();
  widgets.patchValues.mockClear();
  host = document.createElement('div');
  host.style.width = '800px';
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  listeners.clear();
});

describe('native workflow-form accordion', () => {
  it.each([false, true])(
    'preserves side-by-side fields in a headed row (root: %s), including after reopening',
    async (atRoot) => {
      const projectGraph = layoutGraph('row', atRoot);
      await renderForm(16, projectGraph);
      const first = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
      const second = host.querySelector<HTMLInputElement>('input[aria-label="Second value"]')!;
      expect(first).not.toBeNull();
      expect(second).not.toBeNull();
      const expectRow = () => {
        const a = first.getBoundingClientRect();
        const b = second.getBoundingClientRect();
        expect(a.width).toBeGreaterThan(0);
        expect(Math.abs(a.top - b.top)).toBeLessThanOrEqual(1);
        expect(b.left).toBeGreaterThanOrEqual(a.right);
      };
      expectRow();
      const trigger = page.getByRole('button', { name: 'Numeric section' });
      await act(() => userEvent.click(trigger));
      await expect.element(trigger).toHaveAttribute('aria-expanded', 'false');
      await act(() => userEvent.click(trigger));
      await expect.element(trigger).toHaveAttribute('aria-expanded', 'true');
      expectRow();
      expect(first.value).toBe('11');
      expect(second.value).toBe('22');
      if (!atRoot) {
        await expect.element(page.getByText('Before row')).toBeVisible();
        await expect.element(page.getByText('After row')).toBeVisible();
      }
    }
  );

  it('preserves a root row without headings', async () => {
    const projectGraph = layoutGraph('row', true);
    const rootElement = projectGraph.form.elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children = ['first', 'second'];
    }
    await renderForm(16, projectGraph);
    const first = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
    const second = host.querySelector<HTMLInputElement>('input[aria-label="Second value"]')!;
    expect(Math.abs(first.getBoundingClientRect().top - second.getBoundingClientRect().top)).toBeLessThanOrEqual(1);
    expect(second.getBoundingClientRect().left).toBeGreaterThanOrEqual(first.getBoundingClientRect().right);
  });

  it('keeps a column section vertical instead of inheriting row behavior globally', async () => {
    await renderForm(16, layoutGraph('column', true));
    const first = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
    const second = host.querySelector<HTMLInputElement>('input[aria-label="Second value"]')!;
    expect(second.getBoundingClientRect().top).toBeGreaterThanOrEqual(first.getBoundingClientRect().bottom);
    expect(second.getBoundingClientRect().left).toBe(first.getBoundingClientRect().left);
  });

  it('keeps a nested column independent from its parent row and reports descendant errors when closed', async () => {
    const projectGraph = layoutGraph('row');
    const group = projectGraph.form.elements.group!;
    if (group.type === 'container') {
      group.data.children.push('inner');
    }
    Object.assign(projectGraph.form.elements, {
      inner: {
        id: 'inner',
        type: 'container',
        parentId: 'group',
        data: { layout: 'column', children: ['inner-title', 'inner-a', 'inner-b', 'missing'] },
      },
      'inner-title': { id: 'inner-title', type: 'heading', parentId: 'inner', data: { content: 'Inner column' } },
      'inner-a': { id: 'inner-a', type: 'text', parentId: 'inner', data: { content: 'Nested first' } },
      'inner-b': { id: 'inner-b', type: 'text', parentId: 'inner', data: { content: 'Nested second' } },
      missing: { ...structuredClone(graph.form.elements.missing!), parentId: 'inner' },
    });
    await renderForm(12, projectGraph);
    const first = page.getByText('Nested first').element().getBoundingClientRect();
    const second = page.getByText('Nested second').element().getBoundingClientRect();
    expect(second.top).toBeGreaterThanOrEqual(first.bottom);
    const inner = page.getByRole('button', { name: /^Inner column/ });
    await act(() => userEvent.click(inner));
    expect(inner.element().textContent).toContain('Needs attention');
    const numeric = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
    expect(numeric.getBoundingClientRect().width).toBeGreaterThan(0);
    const outer = page.getByRole('button', { name: /^Numeric section/ });
    await act(() => userEvent.click(outer));
    expect(outer.element().textContent).toContain('Needs attention');
  });

  it('toggles equal heading titles by identity without closing the other section', async () => {
    const projectGraph = structuredClone(graph);
    for (const id of ['input', 'output']) {
      const heading = projectGraph.form.elements[id]!;
      if (heading.type === 'heading') {
        heading.data.content = 'Repeated section';
      }
    }
    await renderForm(16, projectGraph);
    const triggers = [...host.querySelectorAll<HTMLButtonElement>('[data-part="trigger"]')].filter(
      (button) => button.textContent?.trim() === 'Repeated section'
    );
    expect(triggers).toHaveLength(2);
    await act(() => userEvent.click(triggers[0]!));
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [projectGraph.id]: { input: false } });
    await expect.poll(() => triggers[0]!.getAttribute('aria-expanded')).toBe('false');
    expect(triggers[1]!.getAttribute('aria-expanded')).toBe('true');
  });

  it('restores expansion separately when two workflows reuse the same heading ids', async () => {
    const other = { ...structuredClone(graph), id: 'other-workflow' };
    await renderForm();
    await act(() => userEvent.click(page.getByRole('button', { name: 'Output section' })));
    await renderForm(16, other);
    await expect.element(page.getByRole('button', { name: 'Output section' })).toHaveAttribute('aria-expanded', 'true');
    await renderForm();
    await expect
      .element(page.getByRole('button', { name: 'Output section' }))
      .toHaveAttribute('aria-expanded', 'false');
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [graph.id]: { output: false } });
  });

  it('keeps long heading labels usable without widening a narrow form', async () => {
    const projectGraph = layoutGraph('column', true);
    const heading = projectGraph.form.elements.heading!;
    const label = 'A long heading that must fit the workflow panel '.repeat(6);
    if (heading.type === 'heading') {
      heading.data.content = label;
    }
    host.style.width = '280px';
    await renderForm(8, projectGraph);
    const trigger = page.getByRole('button', { name: label.trim() });
    await expect.element(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(host.scrollWidth).toBeLessThanOrEqual(host.clientWidth + 1);
    await act(() => userEvent.click(trigger));
    await expect.element(trigger).toHaveAttribute('aria-expanded', 'false');
  });

  it('matches native Generate header styling while retaining a distinct configurable section gap', async () => {
    await renderForm(24, graph, true);
    const native = host.querySelector<HTMLElement>('[data-testid="native-reference"] [data-part="trigger"]')!;
    const workflow = page.getByRole('button', { name: 'Output section' }).element();
    const nativeLabel = native.querySelector('span')!;
    const workflowLabel = workflow.querySelector('span')!;
    for (const property of [
      'fontSize',
      'fontWeight',
      'letterSpacing',
      'textTransform',
      'lineHeight',
      'color',
    ] as const) {
      expect(getComputedStyle(workflowLabel)[property]).toBe(getComputedStyle(nativeLabel)[property]);
    }
    for (const property of ['height', 'paddingInlineStart', 'paddingInlineEnd'] as const) {
      expect(getComputedStyle(workflow)[property]).toBe(getComputedStyle(native)[property]);
    }
    expect(getComputedStyle(host.querySelector<HTMLElement>('.chakra-stack')!).gap).toBe('24px');
    const section = workflow.closest<HTMLElement>('[data-part="root"]')!;
    expect(getComputedStyle(section).borderTopWidth).toBe('0px');
  });

  it('shows the native thin boundary only between two closed column sections, centered in the chosen gap', async () => {
    const projectGraph = layoutGraph('column', true);
    const rootElement = projectGraph.form.elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children.push('boundary', 'additional', 'extra');
    }
    Object.assign(projectGraph.form.elements, {
      boundary: { id: 'boundary', type: 'divider', parentId: 'root' },
      additional: { id: 'additional', type: 'heading', parentId: 'root', data: { content: 'Additional section' } },
      extra: { id: 'extra', type: 'text', parentId: 'root', data: { content: 'Second section content' } },
    });
    await renderForm(24, projectGraph, true);
    const first = page.getByRole('button', { name: 'Numeric section' });
    const second = page.getByRole('button', { name: 'Additional section' });
    const firstRoot = first.element().closest<HTMLElement>('[data-part="root"]')!;
    const secondRoot = second.element().closest<HTMLElement>('[data-part="root"]')!;
    const boundary = () => getComputedStyle(secondRoot, '::before');
    expect(boundary().opacity).toBe('0');
    expect(boundary().borderTopWidth).toBe('1px');
    expect(boundary().top).toBe('-12px');
    expect(boundary().pointerEvents).toBe('none');
    await act(() => userEvent.click(second));
    await expect.element(second).toHaveAttribute('aria-expanded', 'false');
    expect(boundary().opacity).toBe('0');
    await act(() => userEvent.click(first));
    await expect.poll(() => boundary().opacity).toBe('1');
    expect(getComputedStyle(firstRoot, '::before').opacity).toBe('0');
    await act(() => userEvent.click(page.getByRole('button', { name: 'Native section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Native second section' })));
    const nativeRoot = page
      .getByRole('button', { name: 'Native second section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    await expect.poll(() => getComputedStyle(nativeRoot, '::before').opacity).toBe('1');
    expect(boundary().borderTopColor).toBe(getComputedStyle(nativeRoot, '::before').borderTopColor);
    await act(() => userEvent.click(first));
    await expect.poll(() => boundary().opacity).toBe('0');
    await renderForm(24, layoutGraph('row', true));
    const rowRoot = page
      .getByRole('button', { name: 'Numeric section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    expect(getComputedStyle(rowRoot, '::before').content).toBe('none');
  });

  it.each([false, true])(
    'keeps a single boundary across column wrappers (both wrapped: %s), including after moving a heading',
    async (bothWrapped) => {
      const projectGraph = layoutGraph('column');
      const elements = projectGraph.form.elements;
      const rootElement = elements.root!;
      if (rootElement.type === 'container') {
        rootElement.data.children = ['outer', ...(bothWrapped ? ['next'] : ['additional', 'extra'])];
      }
      elements.group!.parentId = 'outer';
      Object.assign(elements, {
        outer: {
          id: 'outer',
          type: 'container',
          parentId: 'root',
          data: { layout: 'column', children: ['group'] },
        },
        next: {
          id: 'next',
          type: 'container',
          parentId: 'root',
          data: { layout: 'column', children: ['additional', 'extra'] },
        },
        additional: {
          id: 'additional',
          type: 'heading',
          parentId: bothWrapped ? 'next' : 'root',
          data: { content: 'Additional section' },
        },
        extra: {
          id: 'extra',
          type: 'text',
          parentId: bothWrapped ? 'next' : 'root',
          data: { content: 'Second section content' },
        },
      });
      await renderForm(24, projectGraph);
      const first = page.getByRole('button', { name: 'Numeric section' });
      const second = page.getByRole('button', { name: 'Additional section' });
      await act(() => userEvent.click(first));
      expect(visibleBoundaries()).toHaveLength(0);
      await act(() => userEvent.click(second));
      await expect.poll(() => visibleBoundaries().length).toBe(1);
      const expectCenteredBoundary = () => {
        const boundary = visibleBoundaries()[0]!;
        const top =
          boundary.getBoundingClientRect().top + Number.parseFloat(getComputedStyle(boundary, '::before').top);
        const midpoint =
          (first.element().getBoundingClientRect().bottom + second.element().getBoundingClientRect().top) / 2;
        expect(Math.abs(top - midpoint)).toBeLessThanOrEqual(1);
      };
      expectCenteredBoundary();
      await act(() => userEvent.click(first));
      await expect.poll(() => visibleBoundaries().length).toBe(0);
      await act(() => userEvent.click(first));
      await expect.poll(() => visibleBoundaries().length).toBe(1);
      await act(() => userEvent.click(second));
      await expect.poll(() => visibleBoundaries().length).toBe(0);
      await act(() => userEvent.click(second));
      await expect.poll(() => visibleBoundaries().length).toBe(1);
      if (rootElement.type === 'container') {
        rootElement.data.layout = 'row';
      }
      await renderForm(24, structuredClone(projectGraph));
      await expect.poll(() => visibleBoundaries().length).toBe(0);
      if (rootElement.type === 'container') {
        rootElement.data.layout = 'column';
        rootElement.data.children.splice(1, 0, 'between');
      }
      elements.between = { id: 'between', type: 'text', parentId: 'root', data: { content: 'Visible interruption' } };
      await renderForm(24, structuredClone(projectGraph));
      await expect.poll(() => visibleBoundaries().length).toBe(0);
      await expect.element(page.getByText('Visible interruption')).toBeVisible();
      if (!bothWrapped && rootElement.type === 'container') {
        rootElement.data.children = ['heading', 'outer', 'additional', 'extra'];
        elements.heading!.parentId = 'root';
        const group = elements.group!;
        if (group.type === 'container') {
          group.data.children = ['first', 'second'];
        }
        await renderForm(24, structuredClone(projectGraph));
        await expect.poll(() => visibleBoundaries().length).toBe(1);
        expectCenteredBoundary();
      }
    }
  );

  it('centers nested boundaries in their own body gap and does not leak through open or multi-item row wrappers', async () => {
    const projectGraph = layoutGraph('column');
    const elements = projectGraph.form.elements;
    const rootElement = elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children = ['outer-heading', 'group', 'next'];
    }
    Object.assign(elements, {
      'outer-heading': { id: 'outer-heading', type: 'heading', parentId: 'root', data: { content: 'Outer section' } },
      next: {
        id: 'next',
        type: 'container',
        parentId: 'root',
        data: { layout: 'column', children: ['additional', 'extra'] },
      },
      additional: { id: 'additional', type: 'heading', parentId: 'next', data: { content: 'Additional section' } },
      extra: { id: 'extra', type: 'text', parentId: 'next', data: { content: 'Second section content' } },
    });
    await renderForm(36, projectGraph);
    await act(() => userEvent.click(page.getByRole('button', { name: 'Numeric section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Additional section' })));
    await expect.poll(() => visibleBoundaries().length).toBe(1);
    const boundary = visibleBoundaries()[0]!;
    const first = page.getByRole('button', { name: 'Numeric section' }).element().getBoundingClientRect();
    const second = page.getByRole('button', { name: 'Additional section' }).element().getBoundingClientRect();
    const top = boundary.getBoundingClientRect().top + Number.parseFloat(getComputedStyle(boundary, '::before').top);
    expect(Math.abs(top - (first.bottom + second.top) / 2)).toBeLessThanOrEqual(1);
    expect(getComputedStyle(boundary, '::before').top).toBe('-6px');
    const outer = page.getByRole('button', { name: 'Outer section' });
    await act(() => userEvent.click(outer));
    await expect.poll(() => visibleBoundaries().length).toBe(0);
    await act(() => userEvent.click(outer));
    await expect.poll(() => visibleBoundaries().length).toBe(1);
    const group = elements.group!;
    if (group.type === 'container') {
      group.data.layout = 'row';
      group.data.children.unshift('before');
    }
    elements.before!.parentId = 'group';
    await renderForm(36, structuredClone(projectGraph));
    await expect.poll(() => visibleBoundaries().length).toBe(0);
    await expect.element(page.getByText('Before row')).toBeVisible();
  });

  it('collapses only fields under the clicked heading, preserving errors and later siblings', async () => {
    await renderForm();
    expect(getComputedStyle(host.querySelector<HTMLElement>('.chakra-stack')!).gap).toBe('8px');
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
