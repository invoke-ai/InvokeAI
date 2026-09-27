import type * as DndKitCoreModule from '@dnd-kit/core';
import type { InvocationTemplate, ProjectGraphState, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/react';
import type { ProjectGraphAction } from '@features/workflow/utility';

import { ChakraProvider } from '@chakra-ui/react';
import { useDroppable } from '@dnd-kit/core';
import { WorkflowUiProvider } from '@features/workflow/react';
import { createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { formEdgeDroppableId } from './formBuilderDnd';
import { FormBuilderTab } from './FormBuilderTab';
import { LinearFormView } from './LinearFormView';
import { PanelModeToggle } from './WorkflowLinearPanel';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Probe renders through a passthrough hook spy; compiler memoization can skip pure helper calls independently of
// component renders.
vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof DndKitCoreModule>();

  return { ...actual, useDroppable: vi.fn(actual.useDroppable) };
});

const entryInput = (
  name: string,
  title: string,
  typeName: string,
  required = true
): InvocationTemplate['inputs'][string] => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required,
  title,
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
});
const entryTemplate: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {
    prompt: entryInput('prompt', 'Prompt', 'StringField'),
    scale: entryInput('scale', 'Scale', 'FloatField', false),
    sizes: {
      ...entryInput('sizes', 'Sizes', 'IntegerField'),
      minimum: 1,
      type: { batch: false, cardinality: 'COLLECTION', name: 'IntegerField' },
    },
    steps: entryInput('steps', 'Steps', 'IntegerField'),
    weight: entryInput('weight', 'Weight', 'FloatField'),
  },
  nodePack: 'invokeai',
  outputType: 'test',
  outputs: {},
  tags: [],
  title: 'Entry',
  type: 'entry',
  useCache: true,
  version: '1.0.0',
};

// File-wide: every describe sees only the `entry` template as loaded; the builder fixtures expose no node fields.
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, status: 'loaded', templates: { entry: entryTemplate } }),
}));

describe('Workflow Linear panel mode toggle', () => {
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

  const renderToggle = async () => {
    const Harness = () => {
      const [mode, setMode] = useState<'view' | 'edit'>('view');
      return <PanelModeToggle mode={mode} onChange={setMode} />;
    };

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    return [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
  };

  const selection = (tabs: HTMLButtonElement[]) => tabs.map((tab) => tab.getAttribute('aria-selected'));

  it('exposes View and Edit as a labelled tablist', async () => {
    const tabs = await renderToggle();

    expect(tabs).toHaveLength(2);
    expect(host.querySelector('[role="tablist"]')?.getAttribute('aria-label')).toBeTruthy();
    expect(selection(tabs)).toEqual(['true', 'false']);
    expect(
      tabs
        .map((tab) => tab.getAttribute('aria-controls'))
        .filter((id): id is string => id !== null)
        .map((id) => document.getElementById(id))
    ).not.toContain(null);
  });

  it('activates View and Edit with pointer and arrow keys', async () => {
    const tabs = await renderToggle();

    await act(() => userEvent.click(tabs[1]!));
    expect(selection(tabs)).toEqual(['false', 'true']);

    // Roving focus: the tablist is one tab stop and arrows move within it.
    tabs[1]?.focus();
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(selection(tabs)).toEqual(['true', 'false']);

    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(selection(tabs)).toEqual(['false', 'true']);
  });
});

/** A second drag after reparenting verifies completion survives dragged-card remounts through DndContext ownership. */
describe('Form builder drag and drop (dnd-kit)', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.width = '480px';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  /** root -> [heading "Field A", divider, container(column, empty)] */
  const buildInitialGraph = (): ProjectGraphState => {
    let doc = createProjectGraph('form-dnd-test');

    doc = projectGraphReducer(doc, { content: 'Field A', elementType: 'heading', type: 'addFormElement' });
    doc = projectGraphReducer(doc, { elementType: 'divider', type: 'addFormElement' });
    doc = projectGraphReducer(doc, { elementType: 'container', layout: 'column', type: 'addFormElement' });

    return doc;
  };

  const Harness = ({ initialGraph }: { initialGraph: ProjectGraphState }) => {
    const [projectGraph, setProjectGraph] = useState(initialGraph);
    const editGraph = useCallback((action: ProjectGraphAction) => {
      setProjectGraph((current) => projectGraphReducer(current, action));
    }, []);
    const adapter = useMemo(
      () =>
        ({
          commands: {
            bindLibraryWorkflow: () => undefined,
            editGraph,
            redo: () => undefined,
            replace: () => undefined,
            undo: () => undefined,
          },
          widgets: { open: () => undefined, patchValues: () => undefined },
        }) as unknown as WorkflowUiAdapter,
      [editGraph]
    );

    return (
      <WorkflowUiProvider adapter={adapter}>
        <FormBuilderTab projectGraph={projectGraph} />
      </WorkflowUiProvider>
    );
  };

  const renderHarness = async (initialGraph: ProjectGraphState = buildInitialGraph()): Promise<void> => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness initialGraph={initialGraph} />
        </ChakraProvider>
      );
    });
  };

  /** The title-bar `HStack` is the drag handle: it's the direct DOM parent of its title `Text`. */
  const titleBarFor = (title: string): HTMLElement => {
    const leaf = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.children.length === 0 && element.textContent?.trim() === title
    );

    if (!leaf?.parentElement) {
      throw new Error(`title bar not found for "${title}"`);
    }

    return leaf.parentElement;
  };

  /** The card's content `Box` — the title bar's rounded-chrome parent's second (and last) child. */
  const cardContentFor = (title: string): HTMLElement => {
    const chrome = titleBarFor(title).parentElement;
    const content = chrome?.lastElementChild;

    if (!(content instanceof HTMLElement)) {
      throw new Error(`card content not found for "${title}"`);
    }

    return content;
  };

  const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
    target.dispatchEvent(
      new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
    );
  };

  const key = (target: EventTarget, code: string): void => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code }));
  };

  // Re-measuring and sensor activation run on rAF outside React's synchronous
  // event handling, so each step needs a real tick — a bare `act()` leaves
  // `over` stale and the drop resolves against the wrong target.
  const interact = (action: () => void): Promise<void> =>
    act(async () => {
      action();
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

  /** Drags `sourceTitle`'s title bar to `(x, y)` with a >4px jitter move first to arm the PointerSensor. */
  const dragTo = async (sourceTitle: string, x: number, y: number): Promise<void> => {
    const handle = titleBarFor(sourceTitle);
    const startRect = handle.getBoundingClientRect();
    const startX = startRect.left + startRect.width / 2;
    const startY = startRect.top + startRect.height / 2;

    await interact(() => pointer('pointerdown', handle, startX, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, startX + 8, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, x, y));
    // The remeasure lands a tick after the move that triggered it, so a no-op
    // settle move re-runs collision detection against current rects.
    await interact(() => pointer('pointermove', handle.ownerDocument, x, y + 1));
    await interact(() => pointer('pointerup', handle.ownerDocument, x, y + 1));
  };

  /** Keyboard drag uses Space/arrows/Space and a net-zero settle nudge for delayed measurement. */
  const dragToWithKeyboard = async (sourceTitle: string, targetCenterY: number): Promise<void> => {
    const handle = titleBarFor(sourceTitle);
    const startRect = handle.getBoundingClientRect();
    const startCenterY = startRect.top + startRect.height / 2;
    const direction = targetCenterY >= startCenterY ? 'ArrowDown' : 'ArrowUp';
    const opposite = direction === 'ArrowDown' ? 'ArrowUp' : 'ArrowDown';
    const steps = Math.round(Math.abs(targetCenterY - startCenterY) / 25);

    handle.focus();
    expect(handle.ownerDocument.activeElement).toBe(handle);

    // Space lifts (dnd-kit's `KeyboardSensor` start code).
    await interact(() => key(handle, 'Space'));

    for (let step = 0; step < steps; step++) {
      await interact(() => key(handle, direction));
    }
    await interact(() => key(handle, direction));
    await interact(() => key(handle, opposite));

    await interact(() => key(handle, 'Space'));
  };

  it('keeps dragging alive after a field is dropped into a container', async () => {
    await renderHarness();

    // Drag 1: "Field A" (a heading) into the empty container's drop zone.
    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();

    await dragTo('Heading', dropZoneRect.left + dropZoneRect.width / 2, dropZoneRect.top + dropZoneRect.height / 2);

    const containerContent = cardContentFor('Container (column)');

    expect(containerContent.textContent).toContain('Heading');
    expect(containerContent.textContent).not.toContain('Empty container');

    // Immediately drag again after reparenting to catch lost completion state from the remounted source.
    const headingCardRect = titleBarFor('Heading').parentElement!.getBoundingClientRect();

    await dragTo('Divider', headingCardRect.left + headingCardRect.width / 2, headingCardRect.bottom - 2);

    // The divider moved into the container, next to the heading.
    const containerContentAfter = cardContentFor('Container (column)');

    expect(containerContentAfter.textContent).toContain('Heading');
    expect(containerContentAfter.textContent).toContain('Divider');

    // No card is left stuck at the mid-drag 40% opacity.
    const opacities = [...host.querySelectorAll<HTMLElement>('*')].map((element) => getComputedStyle(element).opacity);

    expect(opacities).not.toContain('0.4');
  });

  /** Exercise KeyboardSensor and rectangle collision fallback; keyboard drags have no pointer coordinates. */
  it('moves a form element into a container with the keyboard', async () => {
    await renderHarness();

    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();

    await dragToWithKeyboard('Heading', dropZoneRect.top + dropZoneRect.height / 2);

    const containerContent = cardContentFor('Container (column)');

    expect(containerContent.textContent).toContain('Heading');
    expect(containerContent.textContent).not.toContain('Empty container');
  });

  /** Test keyboard edge targeting separately from container drops to exercise the translated-card-center fallback. */
  it('reorders a form element above a sibling with the keyboard', async () => {
    await renderHarness();

    const headingCardRect = titleBarFor('Heading').parentElement!.getBoundingClientRect();

    await dragToWithKeyboard('Divider', headingCardRect.top + headingCardRect.height * 0.25);

    const leafElements = [...host.querySelectorAll<HTMLElement>('*')].filter(
      (element) => element.children.length === 0
    );
    const dividerIndex = leafElements.findIndex((element) => element.textContent?.trim() === 'Divider');
    const headingIndex = leafElements.findIndex((element) => element.textContent?.trim() === 'Heading');

    expect(dividerIndex).toBeGreaterThanOrEqual(0);
    expect(headingIndex).toBeGreaterThanOrEqual(0);
    expect(dividerIndex).toBeLessThan(headingIndex);
  });

  /**
   * Count untouched card hook calls during pointer moves to verify drop-target context isolation despite compiler
   * memoization.
   */
  it('does not re-render an unrelated card on a drag-move that only changes the drop target', async () => {
    const initialGraph = buildInitialGraph();
    const dividerId = Object.values(initialGraph.form.elements).find((element) => element.type === 'divider')!.id;
    const dividerDroppableId = formEdgeDroppableId(dividerId);

    await renderHarness(initialGraph);

    // Capture the empty hint's rect before drag changes its text to Drop here.
    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();
    const midX = dropZoneRect.left + dropZoneRect.width / 2;
    const midY = dropZoneRect.top + dropZoneRect.height / 2;

    const handle = titleBarFor('Heading');
    const startRect = handle.getBoundingClientRect();
    const startX = startRect.left + startRect.width / 2;
    const startY = startRect.top + startRect.height / 2;

    // Activate the sensor and settle delayed measurement before testing moves within one logical target.
    await interact(() => pointer('pointerdown', handle, startX, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, startX + 8, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 1));
    vi.mocked(useDroppable).mockClear();

    // Move within the same target to exercise per-frame state churn without changing the intended drop.
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 2));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 3));

    const dividerCardRerendered = vi
      .mocked(useDroppable)
      .mock.calls.some(([options]) => options.id === dividerDroppableId);

    expect(dividerCardRerendered).toBe(false);

    // End the drag cleanly so it doesn't leak into other tests.
    await interact(() => pointer('pointerup', handle.ownerDocument, midX, midY + 3));
  });
});

describe('Linear form field entry', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.width = '320px';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const entryNode: WorkflowInvocationNode = {
    data: {
      inputs: {
        prompt: { label: '', name: 'prompt', value: 'hello world' },
        scale: { label: '', name: 'scale', value: 1.5 },
        sizes: { label: '', name: 'sizes', value: [4, 8] },
        steps: { label: '', name: 'steps', value: 20 },
        weight: { label: '', name: 'weight', value: 0.5 },
      },
      isIntermediate: true,
      isOpen: true,
      label: '',
      nodePack: 'invokeai',
      notes: '',
      type: 'entry',
      useCache: true,
      version: '1.0.0',
    },
    id: 'entry-node',
    position: { x: 0, y: 0 },
    type: 'invocation',
  };

  /** The entry node with every field exposed on the form. */
  const buildGraph = (): ProjectGraphState => {
    let doc = projectGraphReducer(createProjectGraph('linear-entry-test'), { node: entryNode, type: 'addNode' });

    for (const fieldName of Object.keys(entryTemplate.inputs)) {
      doc = projectGraphReducer(doc, { fieldIdentifier: { fieldName, nodeId: entryNode.id }, type: 'exposeField' });
    }

    return doc;
  };

  const renderForm = async () => {
    let latest = buildGraph();
    const Harness = () => {
      const [projectGraph, setProjectGraph] = useState(latest);
      const editGraph = useCallback((action: ProjectGraphAction) => {
        setProjectGraph((current) => {
          latest = projectGraphReducer(current, action);
          return latest;
        });
      }, []);
      const adapter = useMemo(
        () =>
          ({
            commands: { bindLibraryWorkflow: vi.fn(), editGraph, redo: vi.fn(), replace: vi.fn(), undo: vi.fn() },
            widgets: { open: vi.fn(), patchValues: vi.fn() },
          }) as unknown as WorkflowUiAdapter,
        [editGraph]
      );

      return (
        <WorkflowUiProvider adapter={adapter}>
          <LinearFormView projectGraph={projectGraph} />
        </WorkflowUiProvider>
      );
    };

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const field = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
    const value = (name: string) => {
      const node = latest.nodes.find((candidate) => candidate.id === entryNode.id);
      return node?.type === 'invocation' ? node.data.inputs[name]?.value : undefined;
    };
    const error = (input: HTMLInputElement) =>
      input.closest('[data-scope="field"][data-part="root"]')?.querySelector('[data-part="error-text"]')?.textContent ??
      null;

    return { error, field, value };
  };
  const focusAtEnd = (input: HTMLInputElement) =>
    act(async () => {
      await userEvent.click(input);
      await userEvent.keyboard('{End}');
    });
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });

  it('keeps the caret while editing an exposed text field through the project graph', async () => {
    const { error, field, value } = await renderForm();
    const prompt = field('Prompt');

    await focusAtEnd(prompt);
    await keys('{Home}{ArrowRight}{ArrowRight}XYZ');
    expect(prompt.value).toBe('heXYZllo world');
    expect([prompt.selectionStart, prompt.selectionEnd]).toEqual([5, 5]);
    expect(value('prompt')).toBe('heXYZllo world');
    expect(field('Prompt')).toBe(prompt);

    await keys('{Control>}a{/Control}{Backspace}');
    expect(prompt.value).toBe('');
    expect(value('prompt')).toBe('');
    expect(document.activeElement).toBe(prompt);
    expect(error(prompt)).toBeNull();
  });

  it('commits numeric drafts as typed and reports required, invalid, and optional-empty states', async () => {
    const { error, field, value } = await renderForm();
    const weight = field('Weight');

    await focusAtEnd(weight);
    await keys('0');
    expect(weight.value).toBe('0.50');
    expect(value('weight')).toBe(0.5);
    await act(() => weight.blur());
    expect(weight.value).toBe('0.5');

    await focusAtEnd(weight);
    await keys('{Control>}a{/Control}{Backspace}');
    expect(weight.value).toBe('');
    expect(value('weight')).toBeUndefined();
    expect(document.activeElement).toBe(weight);
    expect(error(weight)).toBe('Required value.');

    await keys('-');
    await keys('2');
    expect(weight.value).toBe('-2');
    expect(value('weight')).toBe(-2);
    expect(error(weight)).toBeNull();

    const steps = field('Steps');

    await focusAtEnd(steps);
    await keys('.5');
    expect(steps.value).toBe('20.5');
    expect(value('steps')).toBe(20.5);
    expect(steps.getAttribute('aria-invalid')).toBe('true');
    expect(error(steps)).toBe('Invalid value.');
    await keys('{Backspace}{Backspace}');
    expect(value('steps')).toBe(20);
    expect(error(steps)).toBeNull();

    const scale = field('Scale');

    await focusAtEnd(scale);
    await keys('{Control>}a{/Control}{Delete}');
    expect(scale.value).toBe('');
    expect(value('scale')).toBeUndefined();
    expect(error(scale)).toBeNull();
    await act(() => scale.blur());
    expect(scale.value).toBe('');
    expect(scale.getAttribute('aria-invalid')).toBeNull();
  });

  it('edits an exposed scalar list row by row and reports the first empty entry', async () => {
    const { value } = await renderForm();
    // Raw i18n keys: every list row shares one accessible name, so rows are found by position.
    const listRow = (index: number) =>
      host.querySelectorAll<HTMLInputElement>('input[aria-label="nodes.collectionItemLabel"]')[index - 1]!;
    // Each row scopes its own Field.Root; the reason text belongs to the host field around the list.
    const listError = (input: HTMLInputElement) =>
      input
        .closest('[data-scope="field"][data-part="root"]')
        ?.parentElement?.closest('[data-scope="field"][data-part="root"]')
        ?.querySelector('[data-part="error-text"]')?.textContent ?? null;

    expect(listRow(2).value).toBe('8');

    const addItem = Array.from(host.querySelectorAll('button')).find(
      (button) => button.textContent === 'nodes.addItem'
    )!;

    await act(async () => {
      await userEvent.click(addItem);
    });
    expect(value('sizes')).toEqual([4, 8, 1]);

    const third = listRow(3);

    await focusAtEnd(third);
    await keys('6');
    expect(third.value).toBe('16');
    expect(value('sizes')).toEqual([4, 8, 16]);
    expect(listError(third)).toBeNull();

    await keys('{Control>}a{/Control}{Backspace}');
    expect(value('sizes')).toEqual([4, 8, null]);
    expect(third.getAttribute('aria-invalid')).toBe('true');
    expect(listError(third)).toBe('Item 3 is empty.');
    // The host field is invalid as a whole, yet only the offending row carries the invalid state.
    expect(listRow(1).getAttribute('aria-invalid')).toBeNull();
    expect(listRow(1).getAttribute('data-invalid')).toBeNull();
    expect(listRow(2).getAttribute('aria-invalid')).toBeNull();
  });
});
