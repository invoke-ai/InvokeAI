import type { InvocationTemplate, ProjectGraphState } from '@features/workflow/contracts';
import type { ProjectGraphAction } from '@features/workflow/core/document';
import type * as workflowReact from '@features/workflow/react';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerateCollapsibleSection } from '@features/generation/ui/shared/GenerateCollapsibleSection';
import { WorkflowUiProvider } from '@features/workflow/react';
import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { FormBuilderTab } from './FormBuilderTab';
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

const TEMPLATES_SNAPSHOT = {
  error: null,
  status: 'loaded' as const,
  templates: { [numericTemplate.type]: numericTemplate },
};

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof workflowReact>()),
  useInvocationTemplatesSelector: (select: (snapshot: unknown) => unknown) => select(TEMPLATES_SNAPSHOT),
}));

vi.mock('@features/generation/ui/GenerationUiContext', () => ({
  useGenerationUi: () => ({ sectionPreferences: { sectionsOpen: {}, setSectionsOpen: vi.fn() } }),
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
          children: ['intro', 'input', 'divider', 'between', 'output', 'tail'],
        },
      },
      intro: { id: 'intro', type: 'text', parentId: 'root', data: { content: 'Always visible' } },
      input: {
        id: 'input',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Input section', children: ['missing'] },
      },
      missing: {
        id: 'missing',
        type: 'node-field',
        parentId: 'input',
        data: {
          fieldIdentifier: { nodeId: 'missing-node', fieldName: 'value' },
          showDescription: false,
          showShuffle: false,
        },
      },
      divider: { id: 'divider', type: 'divider', parentId: 'root' },
      between: { id: 'between', type: 'text', parentId: 'root', data: { content: 'Between sections' } },
      output: {
        id: 'output',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Output section', children: ['result'] },
      },
      result: { id: 'result', type: 'text', parentId: 'output', data: { content: 'A retained value' } },
      tail: {
        id: 'tail',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Final section', children: ['end'] },
      },
      end: { id: 'end', type: 'text', parentId: 'tail', data: { content: 'Final value' } },
    },
  },
};

const layoutGraph = (layout: 'row' | 'column', atRoot = false): ProjectGraphState => {
  const node = { ...buildInvocationNode(numericTemplate, { x: 0, y: 0 }), id: 'numeric-node' };
  const innerParentId = layout === 'row' ? 'row-inner' : 'section';
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
            layout: 'column',
            children: atRoot ? ['section'] : ['before', 'group', 'after'],
          },
        },
        ...(!atRoot
          ? {
              before: { id: 'before', type: 'text' as const, parentId: 'root', data: { content: 'Before row' } },
              group: {
                id: 'group',
                type: 'container' as const,
                parentId: 'root',
                data: { layout: 'column' as const, children: ['section'] },
              },
              after: { id: 'after', type: 'text' as const, parentId: 'root', data: { content: 'After row' } },
            }
          : {}),
        section: {
          id: 'section',
          type: 'collapsible',
          parentId: atRoot ? 'root' : 'group',
          data: {
            content: 'Numeric section',
            children: layout === 'row' ? ['row-inner'] : ['first', 'second'],
          },
        },
        ...(layout === 'row'
          ? {
              'row-inner': {
                id: 'row-inner',
                type: 'container' as const,
                parentId: 'section',
                data: { layout: 'row' as const, children: ['first', 'second'] },
              },
            }
          : {}),
        first: {
          id: 'first',
          type: 'node-field',
          parentId: innerParentId,
          data: {
            fieldIdentifier: { nodeId: node.id, fieldName: 'first' },
            showDescription: false,
            showShuffle: false,
          },
        },
        second: {
          id: 'second',
          type: 'node-field',
          parentId: innerParentId,
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

const renderForm = async (projectGraph = graph, withNativeReference = false) => {
  snapshot = { ...snapshot, projectGraph };
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WorkflowUiProvider adapter={adapter}>
            <LinearFormView projectGraph={projectGraph} />
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

const isTransparentColor = (color: string) =>
  color === 'transparent' || color === 'rgba(0, 0, 0, 0)' || /\/\s*0\)$/.test(color);

const isVisibleHairline = (element: HTMLElement, edge: 'top' | 'bottom' = 'top') => {
  const style = getComputedStyle(element);
  const width = edge === 'top' ? style.borderTopWidth : style.borderBottomWidth;
  const color = edge === 'top' ? style.borderTopColor : style.borderBottomColor;
  return element.getBoundingClientRect().width > 0 && width === '1px' && !isTransparentColor(color);
};

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
    'preserves side-by-side fields in a row container nested inside a collapsible section (root: %s)',
    async (atRoot) => {
      const projectGraph = layoutGraph('row', atRoot);
      await renderForm(projectGraph);
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

  it('preserves a root row without collapsible sections', async () => {
    const projectGraph = layoutGraph('row', true);
    const rootElement = projectGraph.form.elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.layout = 'row';
      rootElement.data.children = ['first', 'second'];
    }
    delete projectGraph.form.elements.section;
    await renderForm(projectGraph);
    const first = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
    const second = host.querySelector<HTMLInputElement>('input[aria-label="Second value"]')!;
    expect(Math.abs(first.getBoundingClientRect().top - second.getBoundingClientRect().top)).toBeLessThanOrEqual(1);
    expect(second.getBoundingClientRect().left).toBeGreaterThanOrEqual(first.getBoundingClientRect().right);
  });

  it('keeps a column section and nested column containers vertical with 10px field spacing', async () => {
    const projectGraph = layoutGraph('column', true);
    const section = projectGraph.form.elements.section!;
    if (section.type === 'collapsible') {
      section.data.children = ['col-inner'];
    }
    projectGraph.form.elements['col-inner'] = {
      id: 'col-inner',
      type: 'container',
      parentId: 'section',
      data: { layout: 'column', children: ['first', 'second'] },
    };
    projectGraph.form.elements.first!.parentId = 'col-inner';
    projectGraph.form.elements.second!.parentId = 'col-inner';
    await renderForm(projectGraph);
    const first = host.querySelector<HTMLInputElement>('input[aria-label="First value"]')!;
    const second = host.querySelector<HTMLInputElement>('input[aria-label="Second value"]')!;
    expect(second.getBoundingClientRect().top).toBeGreaterThanOrEqual(first.getBoundingClientRect().bottom);
    expect(second.getBoundingClientRect().left).toBe(first.getBoundingClientRect().left);
    const firstField = first.closest<HTMLElement>('[data-scope="field"][data-part="root"]')!;
    const secondField = second.closest<HTMLElement>('[data-scope="field"][data-part="root"]')!;
    expect(
      Math.abs(secondField.getBoundingClientRect().top - firstField.getBoundingClientRect().bottom - 10)
    ).toBeLessThanOrEqual(1);
  });

  it('keeps a nested collapsible independent and reports descendant errors when closed', async () => {
    const projectGraph = layoutGraph('row');
    const section = projectGraph.form.elements.section!;
    if (section.type === 'collapsible') {
      section.data.children.push('inner');
    }
    Object.assign(projectGraph.form.elements, {
      inner: {
        id: 'inner',
        type: 'collapsible',
        parentId: 'section',
        data: { content: 'Inner column', children: ['inner-a', 'inner-b', 'missing'] },
      },
      'inner-a': { id: 'inner-a', type: 'text', parentId: 'inner', data: { content: 'Nested first' } },
      'inner-b': { id: 'inner-b', type: 'text', parentId: 'inner', data: { content: 'Nested second' } },
      missing: { ...structuredClone(graph.form.elements.missing!), parentId: 'inner' },
    });
    await renderForm(projectGraph);
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

  it('toggles equal section titles by identity without closing the other section', async () => {
    const projectGraph = structuredClone(graph);
    for (const id of ['input', 'output']) {
      const section = projectGraph.form.elements[id]!;
      if (section.type === 'collapsible') {
        section.data.content = 'Repeated section';
      }
    }
    await renderForm(projectGraph);
    const triggers = [...host.querySelectorAll<HTMLButtonElement>('[data-part="trigger"]')].filter(
      (button) => button.textContent?.trim() === 'Repeated section'
    );
    expect(triggers).toHaveLength(2);
    await act(() => userEvent.click(triggers[0]!));
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [projectGraph.id]: { input: false } });
    await expect.poll(() => triggers[0]!.getAttribute('aria-expanded')).toBe('false');
    expect(triggers[1]!.getAttribute('aria-expanded')).toBe('true');
  });

  it('restores expansion separately when two workflows reuse the same section ids', async () => {
    const other = { ...structuredClone(graph), id: 'other-workflow' };
    await renderForm();
    await act(() => userEvent.click(page.getByRole('button', { name: 'Output section' })));
    await renderForm(other);
    await expect.element(page.getByRole('button', { name: 'Output section' })).toHaveAttribute('aria-expanded', 'true');
    await renderForm();
    await expect
      .element(page.getByRole('button', { name: 'Output section' }))
      .toHaveAttribute('aria-expanded', 'false');
    expect(snapshot.workflowValues.formSectionsOpen).toEqual({ [graph.id]: { output: false } });
  });

  it('keeps long section labels usable without widening a narrow form', async () => {
    const projectGraph = layoutGraph('column', true);
    const section = projectGraph.form.elements.section!;
    const label = 'A long heading that must fit the workflow panel '.repeat(6);
    if (section.type === 'collapsible') {
      section.data.content = label;
    }
    host.style.width = '280px';
    await renderForm(projectGraph);
    const trigger = page.getByRole('button', { name: label.trim() });
    await expect.element(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(host.scrollWidth).toBeLessThanOrEqual(host.clientWidth + 1);
    await act(() => userEvent.click(trigger));
    await expect.element(trigger).toHaveAttribute('aria-expanded', 'false');
  });

  it('matches native Generate header styling and spacing', async () => {
    await renderForm(graph, true);
    const native = host.querySelector<HTMLElement>('[data-testid="native-reference"] [data-part="trigger"]')!;
    const workflow = page.getByRole('button', { name: 'Output section' }).element();
    const nativeChip = native.querySelector<HTMLElement>('[data-section-chip]')!;
    const workflowChip = workflow.querySelector<HTMLElement>('[data-section-chip]')!;
    const nativeLabel = nativeChip.querySelector('span')!;
    const workflowLabel = workflowChip.querySelector('span')!;
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
    for (const property of ['height', 'paddingInlineStart', 'paddingInlineEnd', 'backgroundColor'] as const) {
      expect(getComputedStyle(workflowChip)[property]).toBe(getComputedStyle(nativeChip)[property]);
    }
    const outputSection = workflow.closest<HTMLElement>('[data-part="root"]')!;
    const tailSection = page
      .getByRole('button', { name: 'Final section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    const inputSection = page
      .getByRole('button', { name: /^Input section/ })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    const introText = page.getByText('Always visible').element();
    expect(
      Math.abs(tailSection.getBoundingClientRect().top - outputSection.getBoundingClientRect().bottom)
    ).toBeLessThanOrEqual(1);
    expect(
      Math.abs(inputSection.getBoundingClientRect().top - introText.getBoundingClientRect().bottom - 10)
    ).toBeLessThanOrEqual(1);
    expect(isTransparentColor(getComputedStyle(outputSection).borderTopColor)).toBe(true);
  });

  it('shows the native thin boundary only between two closed column sections', async () => {
    const projectGraph = layoutGraph('column', true);
    const rootElement = projectGraph.form.elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children.push('additional');
    }
    Object.assign(projectGraph.form.elements, {
      additional: {
        id: 'additional',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Additional section', children: ['extra'] },
      },
      extra: { id: 'extra', type: 'text', parentId: 'additional', data: { content: 'Second section content' } },
    });
    await renderForm(projectGraph, true);
    const first = page.getByRole('button', { name: 'Numeric section' });
    const second = page.getByRole('button', { name: 'Additional section' });
    const firstRoot = first.element().closest<HTMLElement>('[data-part="root"]')!;
    const secondRoot = second.element().closest<HTMLElement>('[data-part="root"]')!;
    await expect.poll(() => isVisibleHairline(secondRoot)).toBe(false);
    await act(() => userEvent.click(second));
    await expect.element(second).toHaveAttribute('aria-expanded', 'false');
    await expect.poll(() => isVisibleHairline(secondRoot)).toBe(false);
    await act(() => userEvent.click(first));
    await act(() => userEvent.unhover(first));
    await expect.poll(() => isVisibleHairline(secondRoot)).toBe(true);
    expect(isVisibleHairline(firstRoot, 'bottom')).toBe(false);
    await act(() => userEvent.click(page.getByRole('button', { name: 'Native section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Native second section' })));
    await act(() => userEvent.unhover(page.getByRole('button', { name: 'Native second section' })));
    const nativeRoot = page
      .getByRole('button', { name: 'Native second section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    await expect.poll(() => isVisibleHairline(nativeRoot)).toBe(true);
    await expect
      .poll(() => getComputedStyle(nativeRoot).borderTopColor)
      .toBe(getComputedStyle(secondRoot).borderTopColor);
    await act(() => userEvent.click(first));
    await expect.poll(() => isVisibleHairline(secondRoot)).toBe(false);
  });

  it('keeps a single boundary across layout-only column wrappers and hides it when interrupted', async () => {
    const projectGraph = layoutGraph('column');
    const elements = projectGraph.form.elements;
    const rootElement = elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children = ['group', 'additional'];
    }
    Object.assign(elements, {
      additional: {
        id: 'additional',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Additional section', children: ['extra'] },
      },
      extra: { id: 'extra', type: 'text', parentId: 'additional', data: { content: 'Second section content' } },
    });
    await renderForm(projectGraph);
    const firstRoot = page
      .getByRole('button', { name: 'Numeric section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    const secondRoot = page
      .getByRole('button', { name: 'Additional section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    await act(() => userEvent.click(page.getByRole('button', { name: 'Numeric section' })));
    await act(() => userEvent.unhover(page.getByRole('button', { name: 'Numeric section' })));
    expect(isVisibleHairline(secondRoot)).toBe(false);
    await act(() => userEvent.click(page.getByRole('button', { name: 'Additional section' })));
    await act(() => userEvent.unhover(page.getByRole('button', { name: 'Additional section' })));
    await expect.poll(() => isVisibleHairline(secondRoot)).toBe(true);
    expect(isVisibleHairline(firstRoot, 'bottom')).toBe(false);
    expect(
      Math.abs(secondRoot.getBoundingClientRect().top - firstRoot.getBoundingClientRect().bottom)
    ).toBeLessThanOrEqual(1);
    if (rootElement.type === 'container') {
      rootElement.data.children.splice(1, 0, 'between');
    }
    elements.between = { id: 'between', type: 'text', parentId: 'root', data: { content: 'Visible interruption' } };
    await renderForm(structuredClone(projectGraph));
    const interruptingText = page.getByText('Visible interruption').element();
    await expect
      .poll(() =>
        Math.abs(secondRoot.getBoundingClientRect().top - interruptingText.getBoundingClientRect().bottom - 10)
      )
      .toBeLessThanOrEqual(1);
  });

  it('preserves hairlines and flush spacing around a section inside a single-child row wrapper', async () => {
    const projectGraph = layoutGraph('column');
    const elements = projectGraph.form.elements;
    const group = elements.group!;
    if (group.type === 'container') {
      group.data.layout = 'row';
    }
    const rootElement = elements.root!;
    if (rootElement.type === 'container') {
      rootElement.data.children = ['leading', 'group', 'trailing'];
    }
    Object.assign(elements, {
      leading: {
        id: 'leading',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Leading section', children: ['lead-text'] },
      },
      'lead-text': { id: 'lead-text', type: 'text', parentId: 'leading', data: { content: 'Lead' } },
      trailing: {
        id: 'trailing',
        type: 'collapsible',
        parentId: 'root',
        data: { content: 'Trailing section', children: ['trail-text'] },
      },
      'trail-text': { id: 'trail-text', type: 'text', parentId: 'trailing', data: { content: 'Trail' } },
    });
    await renderForm(projectGraph);
    const leadingRoot = page
      .getByRole('button', { name: 'Leading section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    const middleRoot = page
      .getByRole('button', { name: 'Numeric section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;
    const trailingRoot = page
      .getByRole('button', { name: 'Trailing section' })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!;

    await act(() => userEvent.click(page.getByRole('button', { name: 'Leading section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Numeric section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Trailing section' })));
    await act(() => userEvent.unhover(page.getByRole('button', { name: 'Trailing section' })));

    await expect.poll(() => isVisibleHairline(middleRoot)).toBe(true);
    await expect.poll(() => isVisibleHairline(trailingRoot)).toBe(true);
    expect(isVisibleHairline(leadingRoot, 'bottom')).toBe(false);
    expect(isVisibleHairline(middleRoot, 'bottom')).toBe(false);
    expect(
      Math.abs(middleRoot.getBoundingClientRect().top - leadingRoot.getBoundingClientRect().bottom)
    ).toBeLessThanOrEqual(1);
    expect(
      Math.abs(trailingRoot.getBoundingClientRect().top - middleRoot.getBoundingClientRect().bottom)
    ).toBeLessThanOrEqual(1);
  });

  it('collapses only children of the clicked collapsible section, preserving errors and siblings outside it', async () => {
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

  it('restores the same section state after remount and reopens from the keyboard', async () => {
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

  it('adds a collapsible container in FormBuilderTab and edits its title live in LinearFormView', async () => {
    const initial = createProjectGraph('builder-collapsible-test');
    snapshot = { ...snapshot, projectGraph: initial };

    const BuilderAndView = () => {
      const [current, setCurrent] = useState(initial);
      const editGraph = useCallback((action: ProjectGraphAction) => {
        setCurrent((prev) => {
          const next = projectGraphReducer(prev, action);
          snapshot = { ...snapshot, projectGraph: next };
          return next;
        });
      }, []);
      const editAdapter = useMemo(
        () =>
          ({
            commands: { editGraph, redo: vi.fn(), undo: vi.fn() },
            project,
            widgets,
          }) as unknown as WorkflowUiAdapter,
        [editGraph]
      );

      return (
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <WorkflowUiProvider adapter={editAdapter}>
              <FormBuilderTab projectGraph={current} />
              <div data-testid="live-view">
                <LinearFormView projectGraph={current} />
              </div>
            </WorkflowUiProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    };

    await act(() => {
      root.render(<BuilderAndView />);
    });

    const addBtn = page.getByRole('button', { name: 'Add form element' });
    await act(() => userEvent.click(addBtn));
    const collapsibleItem = page.getByRole('menuitem', { name: 'Collapsible' });
    await act(() => userEvent.click(collapsibleItem));

    const viewHost = host.querySelector<HTMLElement>('[data-testid="live-view"]')!;
    expect(viewHost.querySelector('[data-scope="collapsible"]')).not.toBeNull();

    const titleInput = host.querySelector<HTMLInputElement>('input[aria-label="Section title"]')!;
    expect(titleInput).not.toBeNull();
    await act(() => userEvent.type(titleInput, 'Advanced settings'));
    expect(viewHost.textContent).toContain('Advanced settings');
  });
});
