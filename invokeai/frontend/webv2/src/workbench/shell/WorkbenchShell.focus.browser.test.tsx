/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createWorkbenchFocusController, FocusRegionProvider } from '@workbench/focusRegions';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const shellMocks = vi.hoisted(() => ({
  project: {
    floatingWidgets: {
      'image-map-instance': {
        heightPx: 400,
        mode: 'windowed',
        returnIndex: 0,
        returnRegion: 'center',
        stackOrder: 1,
        widthPx: 500,
        x: 40,
        y: 40,
      },
    },
    id: 'project-1',
    layout: { panels: { isLeftOpen: false, isRightOpen: true } },
    name: 'Test project',
    widgetInstances: {
      'image-map-instance': { createdAt: 0, id: 'image-map-instance', state: { values: {} }, typeId: 'image-map' },
      gallery: { createdAt: 0, id: 'gallery', state: { values: {} }, typeId: 'gallery' },
    },
    widgetRegions: {
      bottom: { activeInstanceId: null, instanceIds: [], isCollapsed: false },
      center: { activeInstanceId: null, instanceIds: [], isCollapsed: false },
      left: { activeInstanceId: null, instanceIds: [], isCollapsed: true },
      right: { activeInstanceId: 'gallery', instanceIds: ['gallery'], isCollapsed: false },
    },
  },
  placementProject: {
    floatingPlacements: { 'image-map-instance': { returnIndex: 0, returnRegion: 'center' } },
    projectId: 'project-1',
    widgetInstances: {
      'image-map-instance': { id: 'image-map-instance', typeId: 'image-map' },
      gallery: { id: 'gallery', typeId: 'gallery' },
    },
    widgetRegions: {
      bottom: { activeInstanceId: null, instanceIds: [] },
      center: { activeInstanceId: null, instanceIds: [] },
      left: { activeInstanceId: null, instanceIds: [] },
      right: { activeInstanceId: 'gallery', instanceIds: ['gallery'] },
    },
  },
  widgets: {
    closeFloating: vi.fn(),
    dockFloating: vi.fn(),
    open: vi.fn(),
    raiseFloating: vi.fn(),
    revealFloating: vi.fn(),
    select: vi.fn(),
    setAlignment: vi.fn(),
    setRegionCollapsed: vi.fn(),
    setRegionSize: vi.fn(),
    toggle: vi.fn(),
  },
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => shellMocks.project.id,
  useActiveProjectSelector: <Selected,>(selector: (project: typeof shellMocks.project) => Selected) =>
    selector(shellMocks.project),
  useWorkbenchCommands: () => ({ notifications: { recordWidgetFailure: vi.fn() }, widgets: shellMocks.widgets }),
}));

vi.mock('@features/gallery/utility', () => ({
  GalleryDragCursor: () => null,
  GalleryDragScope: (props: { children?: ReactNode }) => props.children ?? null,
}));

vi.mock('@workbench/widget-frame', () => ({
  WidgetBar: ({
    side,
    onSelect,
  }: {
    side: 'left' | 'right';
    onSelect: (region: 'left' | 'right', id: string) => void;
  }) =>
    side === 'right' ? (
      <>
        <button data-testid="select-docked-tab" type="button" onClick={() => onSelect('right', 'gallery')}>
          Select docked tab
        </button>
        <div
          data-testid="right-panel-focus-target"
          data-focus-region="right"
          tabIndex={-1}
          style={{ height: '300px', width: '200px' }}
        >
          <div data-hotkey-widget-type-id="gallery">Gallery panel</div>
        </div>
      </>
    ) : null,
}));

vi.mock('@workbench/widget-frame/FloatingWidgetLayer', () => ({ FloatingWidgetLayer: () => null }));
vi.mock('@workbench/widgetDnd', () => ({
  getRegionDropState: () => ({ helperText: '', isActive: false, isAllowed: false }),
  isWidgetDndData: () => false,
  isWidgetInstanceDragData: () => false,
  resolveWidgetDragEnd: () => null,
  widgetCollisionDetection: () => [],
  workbenchAutoScroll: {},
}));
vi.mock('@workbench/widgetPlacementMeta', () => ({
  areWidgetPlacementProjectsEqual: () => true,
  getWidgetPlacementProject: () => shellMocks.placementProject,
}));
vi.mock('@workbench/widgetRegionViewModel', () => ({
  createWidgetRegionViewModelFromState: () => ({ availableItems: [], placedItems: [] }),
  getWidgetRegionItems: () => [],
}));
vi.mock('@workbench/widgetRegistry', () => ({
  getWidgetById: () => undefined,
  getWidgetsForRegion: () => [],
  widgetRegistrationFailures: [],
}));

vi.mock('./BottomPanel', () => ({ BottomPanel: () => null }));
vi.mock('./CenterArea', () => ({ CenterArea: () => null }));
vi.mock('./DocumentTitleProgress', () => ({ DocumentTitleProgress: () => null }));
vi.mock('./notifications', () => ({ WorkbenchNotificationToaster: () => null }));
vi.mock('./Panels', () => ({ LeftPanel: () => null, RightPanel: () => null }));
vi.mock('./PasteMediaRuntime', () => ({ PasteMediaRuntime: () => null }));
vi.mock('./ProjectConflictBanner', () => ({ ProjectConflictBanner: () => null }));
vi.mock('./QueueRecoveryBanner', () => ({ QueueRecoveryBanner: () => null }));
vi.mock('./StatusBar', () => ({ StatusBar: () => null }));
vi.mock('./topbar', () => ({ TopBar: () => null }));

import { WorkbenchShell } from './WorkbenchShell';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  for (const command of Object.values(shellMocks.widgets)) {
    command.mockClear();
  }
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchShell rail focus', () => {
  it('moves focus to a docked panel after selecting its rail tab while a floating window held focus', async () => {
    const controller = createWorkbenchFocusController({
      getProjectId: () => shellMocks.project.id,
      isFloating: (instanceId) =>
        Boolean(
          shellMocks.placementProject.floatingPlacements[
            instanceId as keyof typeof shellMocks.placementProject.floatingPlacements
          ]
        ),
    });
    controller.activate({ instanceId: 'image-map-instance', kind: 'floating' });

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={controller}>
            <WorkbenchShell />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });

    const tab = host!.querySelector<HTMLButtonElement>('[data-testid="select-docked-tab"]')!;
    const panel = host!.querySelector<HTMLElement>('[data-testid="right-panel-focus-target"]')!;
    await userEvent.click(tab);

    expect(shellMocks.widgets.select).toHaveBeenCalledWith({
      projectId: 'project-1',
      region: 'right',
      widgetId: 'gallery',
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(panel));
  });
});
