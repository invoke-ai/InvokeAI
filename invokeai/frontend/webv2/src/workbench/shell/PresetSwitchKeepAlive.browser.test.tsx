import { Box, ChakraProvider } from '@chakra-ui/react';
import { getWorkflowFlowInstance } from '@features/workflow/ui/editor/flowInstanceStore';
import { WorkflowEditorView } from '@features/workflow/ui/editor/WorkflowEditorView';
import {
  clearWorkflowViewports,
  getWorkflowViewport,
  getWorkflowViewportKey,
} from '@features/workflow/ui/editor/workflowViewportStore';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import '@xyflow/react/dist/style.css';

/**
 * Model preset application by replacing region ids while preserving merged widget instances for keep-alive
 * resolution.
 */
const keepAliveMocks = vi.hoisted(() => {
  const icon = () => null;
  // Use settled resources so center icons read synchronously without suspending.
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
  const callNodePosition = { x: 0, y: 0 };
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
      callNodePosition.x = 0;
      callNodePosition.y = 0;
    },
    callNodePosition,
    setActiveInstanceId: (activeInstanceId: string | undefined) => publish(createProject({ activeInstanceId })),
    setCallNodePosition: (position: { x: number; y: number }) => Object.assign(callNodePosition, position),
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
vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useQueueItemProgress: () => null,
}));
vi.mock('@features/workflow/react', async (importOriginal) => {
  const original = await importOriginal<Record<string, unknown>>();
  const callSavedWorkflowTemplate = {
    category: 'workflow',
    classification: 'stable',
    description: '',
    inputs: {},
    nodePack: 'invokeai',
    outputType: 'workflow_return_output',
    outputs: {},
    tags: [],
    title: 'Call Saved Workflow',
    type: 'call_saved_workflow',
    useCache: true,
    version: '1.0.0',
  };
  const templates = { call_saved_workflow: callSavedWorkflowTemplate };

  return {
    ...original,
    ensureInvocationTemplatesLoaded: () => {},
    useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
      selector({ status: 'loaded', templates }),
  };
});
vi.mock('@workbench/focusRegions', () => ({ useFocusRegionProps: () => ({}), useHighlightedRegion: () => null }));
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
vi.mock('@workbench/widget-frame', () => {
  const graph = {
    ...createProjectGraph('workflow-a'),
    nodes: [
      {
        data: {
          callSavedWorkflowStatus: 'ready',
          dynamicInputTemplates: {},
          inputs: { workflow_id: { label: '', name: 'workflow_id', value: 'child-workflow' } },
          isIntermediate: false,
          isOpen: true,
          label: '',
          nodePack: 'invokeai',
          notes: '',
          type: 'call_saved_workflow',
          useCache: true,
          version: '1.0.0',
        },
        id: 'call-saved-workflow',
        position: keepAliveMocks.callNodePosition,
        type: 'invocation',
      },
    ],
  };
  const projectSnapshot = {
    activeWorkflow: { document: graph },
    activeWorkflowId: 'workflow-a',
    galleryValues: {},
    id: 'project-a',
    isWorkflowRunning: false,
    projectGraph: graph,
    workflowValues: {},
    workflows: [{ document: graph }],
  };
  const subscribe = () => () => {};
  /* eslint-disable react-perf/jsx-no-new-object-as-prop -- Stable adapter and runtime fixtures for the mounted editor. */
  const adapter = {
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe },
    commands: {
      addWorkflow: () => 'new-workflow',
      createWorkflow: () => 'new-workflow',
      duplicateWorkflow: () => null,
      editGraph: () => {},
      redo: () => {},
      removeWorkflow: () => {},
      renameWorkflow: () => {},
      selectWorkflow: () => {},
      setWorkflowSource: () => {},
      undo: () => {},
    },
    getProjectGraph: () => graph,
    nodeExecution: {
      get: () => null,
      getOrigin: () => null,
      subscribe,
      subscribeOrigin: subscribe,
    },
    notifications: { error: () => {}, info: () => {}, success: () => {} },
    openAddModels: () => {},
    performance: {
      mark: () => {},
      measure: () => {},
      time: <T,>(_name: string, _source: unknown, callback: () => T) => callback(),
    },
    persistence: {
      getSnapshot: () => ({ error: null, hasLocalRecovery: true, lastSavedAt: null, status: 'saved' }),
      subscribe,
    },
    preferences: {
      getSnapshot: () => ({
        reduceMotion: true,
        themeId: 'classic',
        workflowEdgeStyle: 'curved',
        workflowEdgesBehindNodes: false,
        workflowShowMinimap: false,
        workflowSnapToGrid: false,
        workflowValidateConnections: true,
      }),
      subscribe,
    },
    project: { getSnapshot: () => projectSnapshot, subscribe },
    registerModalHotkeyLayer: () => () => {},
    widgets: { open: () => {}, patchValues: () => {} },
  };
  const runtime = {
    commands: { register: () => () => {} },
    hotkeys: { register: () => () => {} },
    instanceId: 'workflow:center',
    region: 'center',
    typeId: 'workflow',
  } as const;
  /* eslint-enable react-perf/jsx-no-new-object-as-prop */

  const WorkflowEditorWidget = () => (
    <WorkflowUiProvider adapter={adapter as never}>
      <WorkflowEditorView runtime={runtime} />
    </WorkflowUiProvider>
  );

  return {
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
      >
        {instanceId === 'workflow:center' ? <WorkflowEditorWidget /> : null}
      </Box>
    ),
    WidgetSourceLockBadge: () => null,
    useWidgetIntentPreloadProps: () => ({}),
  };
});

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
  clearWorkflowViewports();
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

  it('preserves the live workflow viewport when switching between editor and preview', async () => {
    const { setActiveInstanceId } = await renderCenterArea();

    await setActiveInstanceId('workflow:center');

    const viewportKey = getWorkflowViewportKey('project-a', 'workflow-a', 'workflow:center');

    await vi.waitFor(() => expect(getWorkflowFlowInstance()).not.toBeNull());
    const flow = getWorkflowFlowInstance();

    if (!flow) {
      throw new Error('Expected the production workflow editor flow to mount.');
    }

    await act(async () => {
      await flow.setViewport({ x: 212, y: 140, zoom: 0.72 }, { duration: 0 });
    });

    await vi.waitFor(() => expect(getWorkflowViewport(viewportKey)).toEqual({ x: 212, y: 140, zoom: 0.72 }));
    expect(flow.getViewport()).toEqual({ x: 212, y: 140, zoom: 0.72 });
    expect(host?.querySelectorAll('.react-flow__node')).toHaveLength(1);
    const flowElement = host?.querySelector<HTMLElement>('.react-flow');
    const nodeElement = host?.querySelector<HTMLElement>('.react-flow__node');

    expect(flowElement).not.toBeNull();
    expect(nodeElement).not.toBeNull();
    if (!flowElement || !nodeElement) {
      throw new Error('Expected the single call-workflow node inside React Flow.');
    }
    expect(nodeElement.textContent).toContain('Call Saved Workflow');

    const flowBounds = flowElement.getBoundingClientRect();
    const nodeBounds = nodeElement.getBoundingClientRect();

    expect(nodeBounds.right).toBeGreaterThan(flowBounds.left);
    expect(nodeBounds.left).toBeLessThan(flowBounds.right);
    expect(nodeBounds.bottom).toBeGreaterThan(flowBounds.top);
    expect(nodeBounds.top).toBeLessThan(flowBounds.bottom);

    await setActiveInstanceId('preview');
    await setActiveInstanceId('workflow:center');

    await vi.waitFor(() => expect(getWorkflowFlowInstance()?.getViewport()).toEqual({ x: 212, y: 140, zoom: 0.72 }));

    const restoredFlowElement = host?.querySelector<HTMLElement>('.react-flow');
    const restoredNodeElement = host?.querySelector<HTMLElement>('.react-flow__node');

    expect(restoredFlowElement).not.toBeNull();
    expect(restoredNodeElement?.textContent).toContain('Call Saved Workflow');
  });

  it('restores a viewport that keeps a lone call-workflow node visible', async () => {
    keepAliveMocks.setCallNodePosition({ x: 500, y: 350 });
    const { setActiveInstanceId } = await renderCenterArea();
    const savedViewport = { x: -200, y: -180, zoom: 1 };

    await setActiveInstanceId('workflow:center');

    const viewportKey = getWorkflowViewportKey('project-a', 'workflow-a', 'workflow:center');

    await vi.waitFor(() => expect(getWorkflowFlowInstance()).not.toBeNull());
    const flow = getWorkflowFlowInstance();

    if (!flow) {
      throw new Error('Expected the production workflow editor flow to mount.');
    }

    const flowElement = host?.querySelector<HTMLElement>('.react-flow');
    const nodeElement = host?.querySelector<HTMLElement>('.react-flow__node');

    expect(flowElement).not.toBeNull();
    expect(nodeElement).not.toBeNull();
    if (!flowElement || !nodeElement) {
      throw new Error('Expected the single call-workflow node inside React Flow.');
    }
    expect(nodeElement.textContent).toContain('Call Saved Workflow');

    const flowBounds = flowElement.getBoundingClientRect();
    const defaultNodeBounds = nodeElement.getBoundingClientRect();

    expect(defaultNodeBounds.top).toBeGreaterThanOrEqual(flowBounds.bottom);

    await act(async () => {
      await flow.setViewport(savedViewport, { duration: 0 });
    });

    await vi.waitFor(() => expect(getWorkflowViewport(viewportKey)).toEqual(savedViewport));
    expect(flow.getViewport()).toEqual(savedViewport);
    expect(host?.querySelectorAll('.react-flow__node')).toHaveLength(1);
    const nodeBounds = nodeElement.getBoundingClientRect();

    expect(nodeBounds.right).toBeGreaterThan(flowBounds.left);
    expect(nodeBounds.left).toBeLessThan(flowBounds.right);
    expect(nodeBounds.bottom).toBeGreaterThan(flowBounds.top);
    expect(nodeBounds.top).toBeLessThan(flowBounds.bottom);

    const flowViewportElement = flowElement.querySelector<HTMLElement>('.react-flow__viewport');
    expect(flowViewportElement).not.toBeNull();
    if (!flowViewportElement) {
      throw new Error('Expected the React Flow viewport element.');
    }
    const savedTransform = Array.from(
      new DOMMatrixReadOnly(getComputedStyle(flowViewportElement).transform).toFloat64Array()
    );
    const readTransform = (style: string | null) => {
      const elementStyle = document.createElement('div').style;
      elementStyle.cssText = style ?? '';
      return Array.from(new DOMMatrixReadOnly(elementStyle.transform).toFloat64Array());
    };
    await setActiveInstanceId('preview');

    const visibleFrameTransforms: number[][] = [];
    let shouldSampleVisibleFrames = true;
    let animationFrameId: number | null = null;
    const sampleVisibleFrame = () => {
      if (!shouldSampleVisibleFrames) {
        return;
      }
      if (flowElement.getClientRects().length > 0) {
        visibleFrameTransforms.push(readTransform(flowViewportElement.getAttribute('style')));
      }
      animationFrameId = window.requestAnimationFrame(sampleVisibleFrame);
    };
    animationFrameId = window.requestAnimationFrame(sampleVisibleFrame);

    await setActiveInstanceId('workflow:center');

    await vi.waitFor(() => expect(getWorkflowFlowInstance()?.getViewport()).toEqual(savedViewport));
    await new Promise<void>((resolve) => {
      window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve()));
    });
    shouldSampleVisibleFrames = false;
    if (animationFrameId !== null) {
      window.cancelAnimationFrame(animationFrameId);
    }
    expect(visibleFrameTransforms.length).toBeGreaterThan(0);
    expect(
      visibleFrameTransforms.filter((transform) => transform.some((value, index) => value !== savedTransform[index]))
    ).toEqual([]);

    const restoredFlowElement = host?.querySelector<HTMLElement>('.react-flow');
    const restoredNodeElement = host?.querySelector<HTMLElement>('.react-flow__node');

    expect(restoredFlowElement).not.toBeNull();
    expect(restoredNodeElement?.textContent).toContain('Call Saved Workflow');
    if (!restoredFlowElement || !restoredNodeElement) {
      throw new Error('Expected the call-workflow node after returning to the editor.');
    }

    const restoredFlowBounds = restoredFlowElement.getBoundingClientRect();
    const restoredNodeBounds = restoredNodeElement.getBoundingClientRect();

    expect(restoredNodeBounds.right).toBeGreaterThan(restoredFlowBounds.left);
    expect(restoredNodeBounds.left).toBeLessThan(restoredFlowBounds.right);
    expect(restoredNodeBounds.bottom).toBeGreaterThan(restoredFlowBounds.top);
    expect(restoredNodeBounds.top).toBeLessThan(restoredFlowBounds.bottom);
  });

  it('leaves a hidden widget out of the tab order', async () => {
    const { setActiveInstanceId } = await renderCenterArea();

    await setActiveInstanceId('workflow:center');

    const hiddenCanvas = centerWidget('canvas');

    expect(hiddenCanvas).not.toBeNull();
    // display:none removes hidden views from accessibility, hit testing, and sequential focus.
    expect(hiddenCanvas?.getClientRects().length).toBe(0);
  });

  it('drops a kept instance once another region starts showing it', async () => {
    const { setActiveInstanceId, setRightInstanceId } = await renderCenterArea();

    await setActiveInstanceId('preview');
    await setActiveInstanceId('workflow:center');

    // Kept and hidden while the right rail shows something else.
    expect(centerWidget('preview')).not.toBeNull();

    await setRightInstanceId('preview');

    // A live rail placement must evict the hidden center copy to avoid mounting one instance twice.
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
