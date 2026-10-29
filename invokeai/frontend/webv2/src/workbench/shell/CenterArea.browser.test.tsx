/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { Box, ChakraProvider, HStack, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ComponentType, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

interface CenterAreaTestProject {
  floatingWidgets?: Record<string, { returnIndex: number; returnRegion: string }>;
  id: string;
  invocation: { sourceId: string };
  queue: { items: never[] };
  widgetInstances: Record<string, { id: string; title?: string; typeId: string }>;
  widgetRegions: {
    center: { activeInstanceId: string; instanceIds: string[]; isCollapsed: boolean; sizePx: number };
  };
}

const centerAreaMocks = vi.hoisted(() => {
  const icon = () => null;
  // Use a settled resource so the center icon reads synchronously without suspending.
  const loadedImplementation = () => {
    const promise: Promise<object> & { status?: string; value?: object } = Promise.resolve({});

    promise.status = 'fulfilled';
    promise.value = {};

    return { getStatus: () => 'loaded', load: () => promise, preload: () => {}, retry: () => promise };
  };
  const activeItem = {
    icon,
    id: 'preview-instance',
    instance: { id: 'preview-instance' },
    label: 'Preview',
    status: 'enabled',
    typeId: 'preview',
    widget: {
      implementation: loadedImplementation(),
      manifest: {
        centerPlacement: 'view',
        chrome: { header: 'visible' },
        icon,
      },
      status: 'enabled',
    },
  };

  const project: CenterAreaTestProject = {
    id: 'test-project',
    invocation: { sourceId: 'generate' },
    queue: { items: [] },
    widgetInstances: {},
    widgetRegions: {
      center: { activeInstanceId: activeItem.id, instanceIds: [activeItem.id], isCollapsed: false, sizePx: 0 },
    },
  };

  return {
    activeItem,
    /** Real widget chrome to place in the slots instead of the fixed-size stand-ins. */
    chromeSlots: null as null | {
      actions: ComponentType<{ region: 'center' }>;
      label: ComponentType<{ region: 'center' }>;
    },
    dockFloating: vi.fn(),
    focusFloating: vi.fn(),
    focusRegion: vi.fn(),
    project,
    revealFloating: vi.fn(),
  };
});

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: true, phase: 'ready', user: { is_admin: true, user_id: 'admin-user' } },
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useModelLoads: () => [],
}));
vi.mock('@features/queue/contracts', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getProjectQueueIndicatorState: () => ({ hasOpenQueueWork: false, progressState: null, runningQueueItemId: null }),
}));
vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useQueueItemProgress: () => null,
}));
vi.mock('@workbench/focusRegions', () => ({
  useFocusRegionProps: () => ({}),
  useHighlightedRegion: () => null,
  useWorkbenchFocus: () => ({
    focusFloating: centerAreaMocks.focusFloating,
    focusRegion: centerAreaMocks.focusRegion,
    getTarget: () => null,
  }),
}));
vi.mock('@workbench/widgetRegionViewModel', () => ({
  // The center's placed views are exactly its docked members.
  createWidgetRegionViewModelFromState: ({ regionState }: { regionState: { instanceIds: string[] } }) => ({
    placedItems: regionState.instanceIds.length > 0 ? [centerAreaMocks.activeItem] : [],
  }),
  getWidgetRegionItems: () => [],
  isRequiredCenterView: () => true,
}));
vi.mock('@workbench/widgetRegistry', () => ({
  // Only the floated-view state resolves a widget by type, for its label.
  getWidgetById: (typeId: string) =>
    typeId === 'preview' ? { manifest: { id: 'preview', label: 'Preview' }, status: 'enabled' } : undefined,
  getWidgetsForRegion: () => [],
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => centerAreaMocks.project.id,
  useActiveProjectSelector: (selector: (project: typeof centerAreaMocks.project) => unknown) =>
    selector(centerAreaMocks.project),
  useWorkbenchCommands: () => ({
    widgets: { dockFloating: centerAreaMocks.dockFloating, revealFloating: centerAreaMocks.revealFloating },
  }),
  useWorkbenchSelector: (selector: (snapshot: { backendConnection: { status: string } }) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));
vi.mock('@workbench/widget-frame', () => ({
  // Loading state is not what this suite measures; the real component would
  // suspend on a chunk that does not exist under the mocked registry.
  WidgetIdentityIcon: () => <Box boxSize="3.5" />,
  WidgetChromeSlotById: ({ slot }: { slot: 'actions' | 'label' }) => {
    const Slot = centerAreaMocks.chromeSlots?.[slot];

    if (Slot) {
      // The frame adds the widget's settings and menu buttons after its own actions.
      return slot === 'actions' ? (
        <>
          <Slot region="center" />
          <Box aria-label="Widget settings" as="button" boxSize="6" flexShrink={0} />
          <Box aria-label="Widget menu" as="button" boxSize="6" flexShrink={0} />
        </>
      ) : (
        <Slot region="center" />
      );
    }

    if (slot === 'actions') {
      return <Box data-testid="trailing-actions" flexShrink={0} w="189px" />;
    }

    return (
      <HStack flex="1" minW="0" w="300px">
        <Text flexShrink={0}>Panther and the Flask</Text>
        <Text truncate>a423880b-184a-4910-977e-fe7d7fa9f732.png</Text>
      </HStack>
    );
  },
  WidgetRendererById: () => <Box h="full" w="full" />,
  WidgetSourceLockBadge: () => null,
  useWidgetIntentPreloadProps: () => ({}),
}));

import { CenterArea } from './CenterArea';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { scrollToEnd: 'Scroll to end', scrollToStart: 'Scroll to start' },
        widgets: {
          centerRegion: 'Center view',
          centerViewEmpty: 'No view',
          centerViewLabel: 'Center view: {{label}}',
          floating: {
            centerFloating: '{{label}} is in a floating window',
            destinations: { right: 'right panel' },
            dockTo: 'Dock to {{destination}}',
            showWindow: 'Show window',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (children: ReactNode, width = 571) => {
  host = document.createElement('div');
  host.style.cssText = `display:flex;height:320px;width:${width}px;`;
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{children}</ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('CenterArea chrome layout', () => {
  it('keeps non-shrinking trailing actions inside the center region', async () => {
    await render(<CenterArea />);

    const centerRegion = host?.querySelector<HTMLElement>('section');
    const trailingActions = host?.querySelector<HTMLElement>('[data-testid="trailing-actions"]');

    expect(centerRegion).not.toBeNull();
    expect(trailingActions).not.toBeNull();
    if (!centerRegion || !trailingActions) {
      throw new Error('Expected center chrome geometry.');
    }

    expect(trailingActions.getBoundingClientRect().right).toBeLessThanOrEqual(
      centerRegion.getBoundingClientRect().right
    );
  });

  it('keeps the islands apart and inside the minimum center width, scrolling what does not fit', async () => {
    // The center's own minimum width; both islands hold more than fits beside each other.
    await render(<CenterArea />, 320);
    // Resize observations, and the cues they set, land before the second frame.
    await act(
      () =>
        new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        })
    );

    const centerRegion = host!.querySelector<HTMLElement>('section')!.getBoundingClientRect();
    const [startIsland, endIsland] = [...host!.querySelectorAll<HTMLElement>('[data-center-chrome-island]')];
    const start = startIsland!.getBoundingClientRect();
    const end = endIsland!.getBoundingClientRect();
    const viewMenu = startIsland!.querySelector('button[aria-label^="Center view"]')!.getBoundingClientRect();

    expect(start.left).toBeGreaterThanOrEqual(centerRegion.left);
    expect(start.right).toBeLessThanOrEqual(end.left);
    expect(end.right).toBeLessThanOrEqual(centerRegion.right);
    // The actions give way at the width that leaves the start island its floor: border, row padding, the view menu's
    // 5rem, and the 2.75rem end cue.
    expect(start.width).toBeGreaterThanOrEqual(2 + 4 + 80 + 44);
    expect(viewMenu.width).toBeGreaterThanOrEqual(80);
    // At least the view menu's own 5rem floor stays clear of the cue covering the start island's end.
    expect(viewMenu.left + 80).toBeLessThanOrEqual(
      startIsland!.querySelector('[data-overflow-cue="end"]')!.getBoundingClientRect().left
    );
    // Each island scrolls the rest, cueing the end that hides it.
    for (const island of [startIsland!, endIsland!]) {
      const row = island.firstElementChild!;
      expect(row.scrollWidth).toBeGreaterThan(row.clientWidth);
      expect(island.querySelector('[data-overflow-cue="end"]')).not.toBeNull();
    }
  });
});

describe('CenterArea with its last view floating', () => {
  const dockedCenter = centerAreaMocks.project.widgetRegions.center;

  afterEach(() => {
    centerAreaMocks.project.floatingWidgets = undefined;
    centerAreaMocks.project.widgetInstances = {};
    centerAreaMocks.project.widgetRegions.center = dockedCenter;
    centerAreaMocks.dockFloating.mockClear();
    centerAreaMocks.focusFloating.mockClear();
    centerAreaMocks.focusRegion.mockClear();
    centerAreaMocks.revealFloating.mockClear();
  });

  const floatLastView = () => {
    centerAreaMocks.project.floatingWidgets = { 'preview-instance': { returnIndex: 2, returnRegion: 'right' } };
    centerAreaMocks.project.widgetInstances = {
      'preview-instance': { id: 'preview-instance', typeId: 'preview' },
    };
    // Emptied, and still naming the view that left.
    centerAreaMocks.project.widgetRegions.center = { ...dockedCenter, instanceIds: [] };
  };
  const button = (name: string) =>
    [...host!.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent === name);

  it('says where the view went and offers both ways back, instead of calling it unavailable', async () => {
    floatLastView();
    await render(<CenterArea />);

    // In the first render: the center never shows blank while its view floats.
    expect(host?.textContent).toContain('Preview is in a floating window');
    expect(button('Show window')).not.toBeUndefined();
    expect(host?.textContent).not.toContain('unavailable');

    await act(async () => {
      button('Show window')?.click();
      await Promise.resolve();
    });

    expect(centerAreaMocks.revealFloating).toHaveBeenCalledWith('preview-instance');
    // Showing the window also moves focus into it.
    expect(centerAreaMocks.focusFloating).toHaveBeenCalledWith('preview-instance');
    expect(centerAreaMocks.dockFloating).not.toHaveBeenCalled();

    await act(async () => {
      button('Dock to right panel')?.click();
      await Promise.resolve();
    });

    expect(centerAreaMocks.dockFloating).toHaveBeenCalledWith('preview-instance');
    // Docking from here restores the center view, so focus follows it.
    expect(centerAreaMocks.focusRegion).toHaveBeenCalledWith('center', 'preview');
  });

  it('still reports an empty center whose view is not floating as unavailable', async () => {
    centerAreaMocks.project.widgetRegions.center = { ...dockedCenter, instanceIds: [] };
    await render(<CenterArea />);

    expect(host?.textContent).toContain('common.widgetViewUnavailable');
    expect(button('Show window')).toBeUndefined();
  });
});

describe('CenterArea with the workflow editor chrome at a narrow width', () => {
  afterEach(() => {
    centerAreaMocks.chromeSlots = null;
    centerAreaMocks.activeItem.label = 'Preview';
  });

  const renderWorkflowChrome = async (source?: { libraryWorkflowId: string; revision: number }) => {
    const [{ loadWorkflowWidgetImplementation }, { WorkflowUiProvider }, { QueryClient, QueryClientProvider }] =
      await Promise.all([
        import('@features/workflow/widget'),
        import('@features/workflow/react'),
        import('@tanstack/react-query'),
      ]);
    const implementation = await loadWorkflowWidgetImplementation();
    const document = { id: 'workflow-1', name: 'A workflow whose name is far too long for a narrow center view' };
    const linkedSource = source ? { ...source, ownerUserId: 'another-user' } : undefined;
    const snapshot = {
      activeWorkflow: { document, source: linkedSource },
      activeWorkflowId: document.id,
      id: 'project-1',
      projectGraph: document,
      workflows: [{ document, source: linkedSource }],
    };
    const port = <T,>(value: T) => ({ getSnapshot: () => value, subscribe: () => () => {} });
    const adapter = {
      persistence: port({ error: null, hasLocalRecovery: true, lastSavedAt: null, status: 'saved' }),
      project: port(snapshot),
    } as unknown as Parameters<typeof WorkflowUiProvider>[0]['adapter'];
    const Label = implementation.headerLabel as ComponentType<{ region: 'center' }>;
    const Actions = implementation.headerActions as ComponentType<{ region: 'center' }>;

    centerAreaMocks.chromeSlots = { actions: Actions, label: Label };
    centerAreaMocks.activeItem.label = 'Workflow';
    // The width the project-file journey's default viewport gives the center view.
    await render(
      <QueryClientProvider client={new QueryClient()}>
        <WorkflowUiProvider adapter={adapter}>
          <CenterArea />
        </WorkflowUiProvider>
      </QueryClientProvider>,
      320
    );
  };

  /** Whether a pointer at the element's centre lands on it, rather than on chrome painted over it. */
  const isHitTestable = (element: Element) => {
    const rect = element.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);

    return rect.width > 0 && hit !== null && element.contains(hit);
  };

  it.each([
    ['unlinked', undefined],
    ['linked, with its Update button', { libraryWorkflowId: 'library-1', revision: 2 }],
  ])('keeps the workflow name and every header action clickable when %s', async (_label, source) => {
    await renderWorkflowChrome(source);

    // Untranslated here: the button is the one carrying the workflow's library link.
    const workflowsButton = host!.querySelector('[data-template-link]')?.closest('button');
    const actions = [
      ...host!.querySelectorAll('[data-hotkey-widget-region="center"] > :last-child :is(button, [role="status"])'),
    ];

    expect(workflowsButton).not.toBeNull();
    expect(actions.length).toBe(source ? 6 : 5);
    expect(isHitTestable(workflowsButton!)).toBe(true);
    expect(
      actions.filter((action) => !isHitTestable(action)).map((action) => action.getAttribute('aria-label'))
    ).toEqual([]);
  });
});
