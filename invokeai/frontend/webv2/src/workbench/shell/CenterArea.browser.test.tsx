/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { Box, ChakraProvider, HStack, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
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
    dockFloating: vi.fn(),
    focusFloating: vi.fn(),
    focusRegion: vi.fn(),
    project,
    revealFloating: vi.fn(),
  };
});

vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/contracts', () => ({
  getProjectQueueIndicatorState: () => ({ hasOpenQueueWork: false, progressState: null, runningQueueItemId: null }),
}));
vi.mock('@features/queue/react', () => ({ useQueueItemProgress: () => null }));
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

const render = async (children: ReactNode) => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;height:320px;width:571px;';
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

    expect(host?.textContent).toContain('Preview is in a floating window');
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

    expect(host?.textContent).toContain('unavailable');
    expect(button('Show window')).toBeUndefined();
  });
});
