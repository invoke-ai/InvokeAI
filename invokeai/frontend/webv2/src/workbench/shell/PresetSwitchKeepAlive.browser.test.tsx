import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * A store whose center region is replaced the way `applyLayoutPresetToProject`
 * replaces it: `instanceIds` is rewritten wholesale, so the outgoing widget is
 * no longer in the region at all. The widget instances are merged and survive,
 * which is what the keep-alive set resolves against.
 */
const keepAliveMocks = vi.hoisted(() => {
  const icon = () => null;
  // A settled deferred resource, the shape `use()` can read without
  // suspending — the center view's icon resolves the implementation to decide
  // whether to show a spinner, and these fixtures are always "already loaded".
  const loadedImplementation = () => {
    const promise: Promise<object> & { status?: string; value?: object } = Promise.resolve({});

    promise.status = 'fulfilled';
    promise.value = {};

    return { getStatus: () => 'loaded', load: () => promise, preload: () => {}, retry: () => promise };
  };
  const widgets: Record<string, { implementation: unknown; manifest: unknown; status: string }> = {
    canvas: {
      implementation: loadedImplementation(),
      manifest: { centerPlacement: 'view', chrome: { header: 'visible' }, icon, id: 'canvas' },
      status: 'enabled',
    },
    workflow: {
      implementation: loadedImplementation(),
      manifest: { centerPlacement: 'view', chrome: { header: 'visible' }, icon, id: 'workflow' },
      status: 'enabled',
    },
  };
  const typeIdByInstanceId: Record<string, string> = {
    canvas: 'canvas',
    layers: 'canvas',
    preview: 'workflow',
    'workflow:center': 'workflow',
  };
  const widgetInstances = {
    canvas: { createdAt: 1, id: 'canvas', typeId: 'canvas' },
    layers: { createdAt: 1, id: 'layers', typeId: 'canvas' },
    preview: { createdAt: 1, id: 'preview', typeId: 'workflow' },
    'workflow:center': { createdAt: 1, id: 'workflow:center', typeId: 'workflow' },
  };
  const createProject = ({
    activeInstanceId,
    projectId = 'project-a',
    rightInstanceId = 'layers',
  }: {
    activeInstanceId: string | undefined;
    projectId?: string;
    rightInstanceId?: string;
  }) => ({
    floatingWidgets: [],
    id: projectId,
    invocation: { sourceId: 'generate' },
    queue: { items: [] },
    widgetInstances,
    widgetRegions: {
      center: {
        activeInstanceId: activeInstanceId ?? '',
        instanceIds: activeInstanceId === undefined ? [] : [activeInstanceId],
        isCollapsed: false,
        sizePx: 0,
      },
      right: { activeInstanceId: rightInstanceId, instanceIds: [rightInstanceId], isCollapsed: false, sizePx: 0 },
    },
  });

  let project = createProject({ activeInstanceId: 'canvas' });
  const listeners = new Set<() => void>();
  const publish = (next: ReturnType<typeof createProject>) => {
    project = next;
    for (const listener of listeners) {
      listener();
    }
  };

  return {
    getProject: () => project,
    getWidgetById: (typeId: string) => widgets[typeId],
    reset: () => {
      project = createProject({ activeInstanceId: 'canvas' });
    },
    setActiveInstanceId: (activeInstanceId: string | undefined) => publish(createProject({ activeInstanceId })),
    setProjectId: (projectId: string) => publish(createProject({ activeInstanceId: 'canvas', projectId })),
    setRightInstanceId: (rightInstanceId: string) =>
      publish(createProject({ activeInstanceId: project.widgetRegions.center.activeInstanceId, rightInstanceId })),
    subscribe: (listener: () => void) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
    typeIdByInstanceId,
    widgets,
  };
});

vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/contracts', () => ({
  getProjectQueueIndicatorState: () => ({ hasOpenQueueWork: false, progressState: null, runningQueueItemId: null }),
}));
vi.mock('@features/queue/react', () => ({ useQueueItemProgress: () => null }));
vi.mock('@workbench/focusRegions', () => ({ useFocusRegionProps: () => ({}) }));
vi.mock('@workbench/widgetRegionViewModel', () => ({
  createWidgetRegionViewModelFromState: ({ regionState }: { regionState: { instanceIds: string[] } }) => ({
    placedItems: regionState.instanceIds.map((instanceId) => {
      const typeId = keepAliveMocks.typeIdByInstanceId[instanceId] ?? instanceId;

      return {
        icon: () => null,
        id: instanceId,
        instance: { id: instanceId, typeId },
        label: typeId,
        status: 'enabled',
        typeId,
        widget: keepAliveMocks.widgets[typeId],
      };
    }),
  }),
  getWidgetRegionItems: () => [],
  isRequiredCenterView: () => true,
}));
vi.mock('@workbench/widgetRegistry', () => ({
  getWidgetById: (typeId: string) => keepAliveMocks.getWidgetById(typeId),
  getWidgetsForRegion: () => [],
}));
vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');

  return {
    useActiveProjectId: () => useSyncExternalStore(keepAliveMocks.subscribe, keepAliveMocks.getProject).id,
    useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
      selector(useSyncExternalStore(keepAliveMocks.subscribe, keepAliveMocks.getProject)),
    useWorkbenchCommands: () => ({ widgets: {} }),
    useWorkbenchSelector: (selector: (snapshot: { backendConnection: { status: string } }) => unknown) =>
      selector({ backendConnection: { status: 'connected' } }),
  };
});
vi.mock('@workbench/widget-frame', () => ({
  // Loading state is not what this suite measures; the real component would
  // suspend on a chunk that does not exist under the mocked registry.
  WidgetIdentityIcon: () => <Box boxSize="3.5" />,
  WidgetChromeSlotById: () => null,
  WidgetRendererById: ({ instanceId }: { instanceId: string }) => (
    <Box
      data-hotkey-widget-instance-id={instanceId}
      data-hotkey-widget-type-id={keepAliveMocks.typeIdByInstanceId[instanceId] ?? instanceId}
      h="full"
      w="full"
    />
  ),
  WidgetSourceLockBadge: () => null,
  useWidgetIntentPreloadProps: () => ({}),
}));

import { CenterArea } from './CenterArea';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: {} } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderCenterArea = async () => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;height:320px;width:571px;';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <CenterArea />
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });

  const publish = async (mutate: () => void) => {
    await act(async () => {
      mutate();
      await Promise.resolve();
    });
  };

  return {
    setActiveInstanceId: (instanceId: string | undefined) =>
      publish(() => keepAliveMocks.setActiveInstanceId(instanceId)),
    setProjectId: (projectId: string) => publish(() => keepAliveMocks.setProjectId(projectId)),
    setRightInstanceId: (instanceId: string) => publish(() => keepAliveMocks.setRightInstanceId(instanceId)),
  };
};

const centerWidget = (instanceId: string) =>
  host?.querySelector<HTMLElement>(`[data-hotkey-widget-instance-id="${instanceId}"]`) ?? null;

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
  keepAliveMocks.reset();
});

describe('preset switch keep-alive', () => {
  it('keeps a center widget mounted when the layout switches away and back', async () => {
    const { setActiveInstanceId } = await renderCenterArea();
    const canvasNode = centerWidget('canvas');

    expect(canvasNode).not.toBeNull();
    if (!canvasNode) {
      throw new Error('Expected the canvas center widget to mount.');
    }

    await setActiveInstanceId('workflow:center');

    // Still in the DOM, still the same element, and hidden rather than destroyed.
    expect(centerWidget('canvas')).toBe(canvasNode);
    expect(getComputedStyle(canvasNode).display).toBe('none');
    expect(getComputedStyle(centerWidget('workflow:center') as Element).display).not.toBe('none');

    await setActiveInstanceId('canvas');

    expect(centerWidget('canvas')).toBe(canvasNode);
    expect(getComputedStyle(canvasNode).display).not.toBe('none');
    expect(getComputedStyle(centerWidget('workflow:center') as Element).display).toBe('none');
  });

  it('leaves a hidden widget out of the tab order', async () => {
    const { setActiveInstanceId } = await renderCenterArea();

    await setActiveInstanceId('workflow:center');

    const hiddenCanvas = centerWidget('canvas');

    expect(hiddenCanvas).not.toBeNull();
    // `display: none` takes the subtree out of the accessibility tree, so it is
    // unreachable by pointer hit-testing and by sequential focus navigation.
    expect(hiddenCanvas?.getClientRects().length).toBe(0);
  });

  it('drops a kept instance once another region starts showing it', async () => {
    const { setActiveInstanceId, setRightInstanceId } = await renderCenterArea();

    await setActiveInstanceId('preview');
    await setActiveInstanceId('workflow:center');

    // Kept and hidden while the right rail shows something else.
    expect(centerWidget('preview')).not.toBeNull();

    await setRightInstanceId('preview');

    // The right rail is now rendering this instance for real; a hidden centre
    // copy would be the same instance mounted twice, one of them a ghost.
    expect(centerWidget('preview')).toBeNull();
  });

  it('forgets what the previous project had shown', async () => {
    const { setActiveInstanceId, setProjectId } = await renderCenterArea();

    await setActiveInstanceId('workflow:center');
    expect(centerWidget('canvas')).not.toBeNull();

    await setProjectId('project-b');

    // Instance ids repeat across projects, so a kept id would resolve to a real
    // instance of the new project carrying the old project's local state.
    expect(centerWidget('workflow:center')).toBeNull();
    expect(centerWidget('canvas')).not.toBeNull();
  });

  it('says the centre is unavailable rather than rendering nothing', async () => {
    const { setActiveInstanceId } = await renderCenterArea();

    await setActiveInstanceId(undefined);

    expect(centerWidget('canvas')).toBeNull();
    expect(host?.textContent).toContain('Center widget unavailable');
  });

  it('keeps chrome on the live region rather than on the kept widget', async () => {
    const { setActiveInstanceId } = await renderCenterArea();

    await setActiveInstanceId('workflow:center');

    const chrome = host?.querySelector<HTMLElement>('[data-hotkey-widget-region="center"]');

    expect(chrome?.dataset.hotkeyWidgetInstanceId).toBe('workflow:center');
  });
});
