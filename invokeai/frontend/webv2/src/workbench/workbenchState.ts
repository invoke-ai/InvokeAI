import {
  defaultInvocationRoute,
  isInvocationRouteValid,
  isInvocationSourceAvailable,
  resolveInvocationRoute,
} from './invocation';
import { defaultLayoutPreset, getLayoutPreset } from './layoutPresets';
import type {
  CanvasStateContract,
  CenterViewId,
  GraphContract,
  GraphHistorySnapshot,
  InvocationRoute,
  InvocationSourceId,
  LayoutPresetId,
  Project,
  ProjectLayoutState,
  ProjectUndoSnapshot,
  QueueItem,
  ResultDestination,
  WidgetFailure,
  WidgetId,
  WidgetRegion,
  WidgetRegionState,
  WidgetStateContract,
  WorkbenchState,
} from './types';

type WorkbenchAction =
  | { type: 'createProject' }
  | { type: 'closeProject'; projectId: string }
  | { type: 'switchProject'; projectId: string }
  | { type: 'setCenterView'; centerViewId: CenterViewId }
  | { type: 'applyPreset'; presetId: LayoutPresetId }
  | { type: 'resetActiveLayout' }
  | { type: 'recoverShellLayout' }
  | { type: 'setInvocationSource'; sourceId: InvocationSourceId }
  | { type: 'setInvocationDestination'; destination: ResultDestination }
  | { type: 'toggleSourceLock' }
  | { type: 'toggleDestinationLock' }
  | { type: 'selectRegionWidget'; region: WidgetRegion; widgetId: WidgetId }
  | { type: 'toggleRegionWidget'; region: WidgetRegion; widgetId: WidgetId }
  | { type: 'setRegionWidgetCollapsed'; region: WidgetRegion; isCollapsed: boolean }
  | { type: 'setRegionWidgetSize'; region: WidgetRegion; sizePx: number }
  | { type: 'submitInvocationSnapshot'; backendSupportsCancellation: boolean }
  | { type: 'submitResolvedInvocationSnapshot'; backendSupportsCancellation: boolean; route: InvocationRoute }
  | { type: 'cancelQueueItem'; queueItemId: string }
  | { type: 'undoProjectChange' }
  | { type: 'redoProjectChange' }
  | { type: 'hydrateWorkbench'; state: WorkbenchState }
  | { type: 'autosaveStarted' }
  | { type: 'autosaveSucceeded'; savedAt: string }
  | { type: 'autosaveFailed'; error: string }
  | { type: 'recordWidgetFailure'; failure: WidgetFailure }
  | { type: 'recordError'; message: string };

const HISTORY_LIMIT = 40;
const INITIAL_PROJECT_COUNT = 3;
const ERROR_LOG_LIMIT = 5;
const MIN_PANEL_SIZE_PX = 180;
const MAX_PANEL_SIZE_PX = 520;
const MIN_STATUS_PANEL_SIZE_PX = 96;
const MAX_STATUS_PANEL_SIZE_PX = 420;

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const cloneGraph = (graph: GraphContract): GraphContract => ({
  ...graph,
  edges: graph.edges.map((edge) => ({ ...edge })),
  nodes: graph.nodes.map((node) => ({ ...node, inputs: { ...node.inputs } })),
});

const cloneCanvas = (canvas: CanvasStateContract): CanvasStateContract => ({
  ...canvas,
  layers: [...canvas.layers],
  stagingArea: {
    ...canvas.stagingArea,
    pendingImageIds: [...canvas.stagingArea.pendingImageIds],
  },
});

const cloneWidgetState = (widgetState: WidgetStateContract): WidgetStateContract => ({
  ...widgetState,
  values: { ...widgetState.values },
});

const cloneWidgetStates = (
  widgetStates: Record<WidgetId, WidgetStateContract>
): Record<WidgetId, WidgetStateContract> => ({
  'autosave-status': cloneWidgetState(widgetStates['autosave-status']),
  canvas: cloneWidgetState(widgetStates.canvas),
  gallery: cloneWidgetState(widgetStates.gallery),
  generate: cloneWidgetState(widgetStates.generate),
  'history-controls': cloneWidgetState(widgetStates['history-controls']),
  'layout-actions': cloneWidgetState(widgetStates['layout-actions']),
  layers: cloneWidgetState(widgetStates.layers),
  queue: cloneWidgetState(widgetStates.queue),
  'server-status': cloneWidgetState(widgetStates['server-status']),
  'version-status': cloneWidgetState(widgetStates['version-status']),
  workflow: cloneWidgetState(widgetStates.workflow),
});

const cloneWidgetRegions = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetRegion, WidgetRegionState> => ({
  center: {
    ...widgetRegions.center,
    enabledWidgetIds: [...widgetRegions.center.enabledWidgetIds],
  },
  left: {
    ...widgetRegions.left,
    enabledWidgetIds: [...widgetRegions.left.enabledWidgetIds],
  },
  right: {
    ...widgetRegions.right,
    enabledWidgetIds: [...widgetRegions.right.enabledWidgetIds],
  },
  bottom: {
    ...widgetRegions.bottom,
    enabledWidgetIds: [...widgetRegions.bottom.enabledWidgetIds],
  },
});

const cloneWidgetGraphs = (widgetGraphs: Project['widgetGraphs']): Project['widgetGraphs'] =>
  Object.fromEntries(Object.entries(widgetGraphs).map(([key, graph]) => [key, graph ? cloneGraph(graph) : graph]));

const createUndoSnapshot = (project: Project): ProjectUndoSnapshot => ({
  canvas: cloneCanvas(project.canvas),
  invocation: { ...project.invocation },
  layout: { ...project.layout, panels: { ...project.layout.panels } },
  projectGraph: cloneGraph(project.projectGraph),
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetRegions: cloneWidgetRegions(project.widgetRegions),
  widgetStates: cloneWidgetStates(project.widgetStates),
});

const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot): Project => ({
  ...project,
  canvas: cloneCanvas(snapshot.canvas),
  invocation: { ...snapshot.invocation },
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  projectGraph: cloneGraph(snapshot.projectGraph),
  widgetGraphs: cloneWidgetGraphs(snapshot.widgetGraphs),
  widgetRegions: cloneWidgetRegions(snapshot.widgetRegions),
  widgetStates: cloneWidgetStates(snapshot.widgetStates),
});

const createGraphHistorySnapshot = (label: string, graph: GraphContract): GraphHistorySnapshot => ({
  createdAt: now(),
  graph: cloneGraph(graph),
  id: createId('graph-history'),
  label,
});

const pushUndo = (project: Project, label: string): Project => ({
  ...project,
  undoRedo: {
    future: [],
    past: [
      ...project.undoRedo.past,
      {
        createdAt: now(),
        id: createId('undo'),
        label,
        project: createUndoSnapshot(project),
      },
    ].slice(-HISTORY_LIMIT),
  },
});

const createProjectGraph = (index: number, projectId: string): GraphContract => ({
  edges: [],
  id: `${projectId}-graph`,
  label: `Project ${index} Graph`,
  nodes: [],
  updatedAt: now(),
  version: 1,
});

const createWidgetStates = (): Record<WidgetId, WidgetStateContract> => ({
  'autosave-status': { id: 'autosave-status', label: 'Autosave', values: {}, version: 1 },
  canvas: { id: 'canvas', label: 'Canvas', values: {}, version: 1 },
  gallery: { id: 'gallery', label: 'Gallery', values: {}, version: 1 },
  generate: { graphId: 'generate-graph', id: 'generate', label: 'Generate', values: {}, version: 1 },
  'history-controls': { id: 'history-controls', label: 'History Controls', values: {}, version: 1 },
  'layout-actions': { id: 'layout-actions', label: 'Layout Actions', values: {}, version: 1 },
  layers: { id: 'layers', label: 'Layers', values: {}, version: 1 },
  queue: { id: 'queue', label: 'Queue', values: {}, version: 1 },
  'server-status': { id: 'server-status', label: 'Server Status', values: {}, version: 1 },
  'version-status': { id: 'version-status', label: 'Version', values: {}, version: 1 },
  workflow: { graphId: 'workflow-graph', id: 'workflow', label: 'Workflow', values: {}, version: 1 },
});

const createWidgetRegions = (): Record<WidgetRegion, WidgetRegionState> => ({
  left: {
    activeWidgetId: 'generate',
    enabledWidgetIds: ['generate', 'workflow'],
    isCollapsed: false,
    sizePx: 288,
  },
  right: {
    activeWidgetId: 'layers',
    enabledWidgetIds: ['queue', 'gallery', 'layers'],
    isCollapsed: false,
    sizePx: 240,
  },
  bottom: {
    activeWidgetId: 'queue',
    enabledWidgetIds: [
      'server-status',
      'queue',
      'gallery',
      'autosave-status',
      'history-controls',
      'layout-actions',
      'version-status',
    ],
    isCollapsed: true,
    sizePx: 180,
  },
  center: {
    activeWidgetId: 'canvas',
    enabledWidgetIds: ['canvas', 'gallery', 'workflow'],
    isCollapsed: false,
    sizePx: 0,
  },
});

const getCenterWidgetIdFromViewId = (centerViewId: CenterViewId): WidgetId =>
  centerViewId === 'preview' ? 'gallery' : centerViewId;

const ensureCenterRegion = (
  centerRegion: WidgetRegionState | undefined,
  fallbackCenterViewId: CenterViewId
): WidgetRegionState => {
  const defaultCenterRegion = createWidgetRegions().center;
  const activeWidgetId = centerRegion?.activeWidgetId ?? getCenterWidgetIdFromViewId(fallbackCenterViewId);
  const enabledWidgetIds = centerRegion?.enabledWidgetIds.length
    ? centerRegion.enabledWidgetIds
    : defaultCenterRegion.enabledWidgetIds;
  const normalizedActiveWidgetId = enabledWidgetIds.includes(activeWidgetId) ? activeWidgetId : enabledWidgetIds[0];

  return {
    ...defaultCenterRegion,
    ...centerRegion,
    activeWidgetId: normalizedActiveWidgetId,
    enabledWidgetIds,
    isCollapsed: false,
  };
};

const ensureProjectWidgetContracts = (project: Project): Project => {
  const defaultWidgetRegions = createWidgetRegions();
  const defaultWidgetStates = createWidgetStates();
  const legacyWidgetRegions = project.widgetRegions as
    | Partial<Record<WidgetRegion | 'left-panel' | 'right-panel' | 'status-bar', WidgetRegionState>>
    | undefined;

  return {
    ...project,
    widgetRegions: {
      left: legacyWidgetRegions?.left ?? legacyWidgetRegions?.['left-panel'] ?? defaultWidgetRegions.left,
      right: legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel'] ?? defaultWidgetRegions.right,
      bottom: legacyWidgetRegions?.bottom ?? legacyWidgetRegions?.['status-bar'] ?? defaultWidgetRegions.bottom,
      center: ensureCenterRegion(legacyWidgetRegions?.center, project.layout.centerViewId),
    },
    widgetStates: {
      ...defaultWidgetStates,
      ...project.widgetStates,
    },
  };
};

const clampPanelSize = (region: WidgetRegion, sizePx: number): number => {
  if (region === 'bottom') {
    return Math.min(MAX_STATUS_PANEL_SIZE_PX, Math.max(MIN_STATUS_PANEL_SIZE_PX, sizePx));
  }

  return Math.min(MAX_PANEL_SIZE_PX, Math.max(MIN_PANEL_SIZE_PX, sizePx));
};

const createCanvasState = (): CanvasStateContract => ({
  layers: [],
  stagingArea: { pendingImageIds: [] },
  version: 1,
});

const createProject = (index: number, id = `project-${index}`): Project => ({
  canvas: createCanvasState(),
  events: [
    {
      createdAt: now(),
      id: createId('event'),
      summary: `Created Project Name #${index}`,
      type: 'project-created',
    },
  ],
  graphHistory: [],
  id,
  invocation: { ...defaultInvocationRoute },
  layout: { ...defaultLayoutPreset.initialLayout, panels: { ...defaultLayoutPreset.initialLayout.panels } },
  name: `Project Name #${index}`,
  projectGraph: createProjectGraph(index, id),
  queue: { items: [] },
  undoRedo: { future: [], past: [] },
  widgetGraphs: {},
  widgetRegions: createWidgetRegions(),
  widgetStates: createWidgetStates(),
});

const getNextProjectIndex = (projects: Project[]): number => {
  const usedIndices = projects.map((project) => Number(project.name.match(/#(\d+)$/)?.[1] ?? 0));

  return Math.max(0, ...usedIndices) + 1;
};

const updateActiveProject = (state: WorkbenchState, getProject: (project: Project) => Project): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) =>
    project.id === state.activeProjectId ? getProject(ensureProjectWidgetContracts(project)) : project
  ),
});

const getNextEnabledWidgetId = (region: WidgetRegionState, widgetId: WidgetId): WidgetId | null => {
  if (region.activeWidgetId !== widgetId) {
    return region.activeWidgetId;
  }

  return region.enabledWidgetIds.find((enabledWidgetId) => enabledWidgetId !== widgetId) ?? null;
};

const updateActiveWidgetRegion = (
  state: WorkbenchState,
  region: WidgetRegion,
  getRegion: (regionState: WidgetRegionState) => WidgetRegionState
): WorkbenchState =>
  updateActiveProject(state, (project) => ({
    ...project,
    widgetRegions: {
      ...project.widgetRegions,
      [region]: getRegion(project.widgetRegions[region]),
    },
  }));

const openPanelForRegion = (layout: ProjectLayoutState, region: WidgetRegion): ProjectLayoutState => ({
  ...layout,
  panels: {
    ...layout.panels,
    isBottomOpen: region === 'bottom' ? true : layout.panels.isBottomOpen,
    isLeftOpen: region === 'left' ? true : layout.panels.isLeftOpen,
    isRightOpen: region === 'right' ? true : layout.panels.isRightOpen,
  },
});

const normalizeWorkbenchState = (state: WorkbenchState): WorkbenchState => ({
  ...state,
  projects: state.projects.map(ensureProjectWidgetContracts),
});

const updateActiveLayout = (
  state: WorkbenchState,
  getLayout: (layout: ProjectLayoutState) => ProjectLayoutState
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');

    return {
      ...nextProject,
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated active layout',
          type: 'layout-updated',
        },
        ...nextProject.events,
      ],
      layout: getLayout(project.layout),
    };
  });

const updateActiveInvocation = (
  state: WorkbenchState,
  getInvocation: (invocation: InvocationRoute) => InvocationRoute
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update invocation route');

    return {
      ...nextProject,
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated invocation source or destination',
          type: 'invocation-updated',
        },
        ...nextProject.events,
      ],
      invocation: getInvocation(project.invocation),
    };
  });

const selectGraphForInvocation = (project: Project, invocation: InvocationRoute): GraphContract => {
  const widgetGraph = project.widgetGraphs[invocation.sourceId as WidgetId];

  return widgetGraph ? cloneGraph(widgetGraph) : cloneGraph(project.projectGraph);
};

const submitInvocationSnapshot = (
  project: Project,
  backendSupportsCancellation: boolean,
  route = resolveInvocationRoute(project)
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const submittedAt = now();
  const queueItemId = createId('queue-item');
  const graph = selectGraphForInvocation(project, route);
  const graphHistorySnapshot = createGraphHistorySnapshot(`Queue snapshot ${queueItemId}`, graph);
  const queueItem: QueueItem = {
    cancellable: backendSupportsCancellation,
    id: queueItemId,
    snapshot: {
      canvas: cloneCanvas(project.canvas),
      destination: route.destination,
      graph,
      sourceId: route.sourceId,
      submittedAt,
      widgetStates: cloneWidgetStates(project.widgetStates),
    },
    status: 'pending',
  };

  return {
    ...project,
    events: [
      {
        createdAt: submittedAt,
        id: createId('event'),
        runId: queueItemId,
        summary: `Submitted immutable ${route.sourceId} graph snapshot to ${route.destination}`,
        type: 'queue-submitted',
      },
      ...project.events,
    ],
    graphHistory: [graphHistorySnapshot, ...project.graphHistory].slice(0, HISTORY_LIMIT),
    invocation: {
      ...project.invocation,
      destination: route.destination,
      lastSubmittedRunId: queueItemId,
      sourceId: route.sourceId,
    },
    queue: { items: [queueItem, ...project.queue.items] },
  };
};

export const createInitialWorkbenchState = (): WorkbenchState => ({
  account: { activeLayoutPresetId: 'canvas-default' },
  activeProjectId: 'project-1',
  autosave: { status: 'idle' },
  errorLog: [],
  projects: Array.from({ length: INITIAL_PROJECT_COUNT }, (_value, index) => createProject(index + 1)),
  widgetFailures: [],
});

export const workbenchReducer = (state: WorkbenchState, action: WorkbenchAction): WorkbenchState => {
  switch (action.type) {
    case 'createProject': {
      const project = createProject(getNextProjectIndex(state.projects), createId('project'));

      return { ...state, activeProjectId: project.id, projects: [...state.projects, project] };
    }
    case 'closeProject': {
      if (state.projects.length === 1) {
        return {
          ...state,
          errorLog: ['At least one project must remain open.', ...state.errorLog],
        };
      }

      const projectIndex = state.projects.findIndex((project) => project.id === action.projectId);
      const projects = state.projects.filter((project) => project.id !== action.projectId);

      if (action.projectId !== state.activeProjectId) {
        return { ...state, projects };
      }

      const fallbackProject = projects[Math.max(0, projectIndex - 1)];

      return { ...state, activeProjectId: fallbackProject.id, projects };
    }
    case 'switchProject': {
      return { ...state, activeProjectId: action.projectId };
    }
    case 'setCenterView': {
      const widgetId = getCenterWidgetIdFromViewId(action.centerViewId);

      return updateActiveWidgetRegion(state, 'center', (region) => ({
        ...region,
        activeWidgetId: region.enabledWidgetIds.includes(widgetId) ? widgetId : region.activeWidgetId,
        isCollapsed: false,
      }));
    }
    case 'applyPreset': {
      const preset = getLayoutPreset(action.presetId);
      const widgetId = getCenterWidgetIdFromViewId(preset.initialLayout.centerViewId);
      const nextState = updateActiveLayout(state, () => ({
        ...preset.initialLayout,
        panels: { ...preset.initialLayout.panels },
      }));

      return {
        ...updateActiveWidgetRegion(nextState, 'center', (region) => ({
          ...region,
          activeWidgetId: region.enabledWidgetIds.includes(widgetId) ? widgetId : region.activeWidgetId,
          isCollapsed: false,
        })),
        account: { activeLayoutPresetId: action.presetId },
      };
    }
    case 'resetActiveLayout': {
      const preset = getLayoutPreset(
        state.projects.find((project) => project.id === state.activeProjectId)?.layout.presetId ??
          state.account.activeLayoutPresetId
      );
      const widgetId = getCenterWidgetIdFromViewId(preset.initialLayout.centerViewId);
      const nextState = updateActiveLayout(state, () => {
        return { ...preset.initialLayout, panels: { ...preset.initialLayout.panels } };
      });

      return updateActiveWidgetRegion(nextState, 'center', (region) => ({
        ...region,
        activeWidgetId: region.enabledWidgetIds.includes(widgetId) ? widgetId : region.activeWidgetId,
        isCollapsed: false,
      }));
    }
    case 'recoverShellLayout': {
      return updateActiveLayout(state, (layout) => ({
        ...layout,
        panels: { isLeftOpen: true, isRightOpen: true, isBottomOpen: true },
      }));
    }
    case 'setInvocationSource': {
      if (!isInvocationSourceAvailable(action.sourceId)) {
        return state;
      }

      return updateActiveInvocation(state, (invocation) => ({ ...invocation, sourceId: action.sourceId }));
    }
    case 'setInvocationDestination': {
      return updateActiveInvocation(state, (invocation) => ({ ...invocation, destination: action.destination }));
    }
    case 'toggleSourceLock': {
      return updateActiveInvocation(state, (invocation) => ({ ...invocation, sourceLocked: !invocation.sourceLocked }));
    }
    case 'toggleDestinationLock': {
      return updateActiveInvocation(state, (invocation) => ({
        ...invocation,
        destinationLocked: !invocation.destinationLocked,
      }));
    }
    case 'selectRegionWidget': {
      return updateActiveProject(state, (project) => {
        const region = project.widgetRegions[action.region];

        if (action.region === 'center') {
          return {
            ...project,
            widgetRegions: {
              ...project.widgetRegions,
              center: { ...region, activeWidgetId: action.widgetId, isCollapsed: false },
            },
          };
        }

        const widgetRegion =
          region.activeWidgetId === action.widgetId
            ? { ...region, isCollapsed: !region.isCollapsed }
            : { ...region, activeWidgetId: action.widgetId, isCollapsed: false };

        return {
          ...project,
          layout: openPanelForRegion(project.layout, action.region),
          widgetRegions: { ...project.widgetRegions, [action.region]: widgetRegion },
        };
      });
    }
    case 'toggleRegionWidget': {
      return updateActiveWidgetRegion(state, action.region, (region) => {
        const isEnabled = region.enabledWidgetIds.includes(action.widgetId);

        if (action.region === 'center' && isEnabled && region.enabledWidgetIds.length === 1) {
          return region;
        }

        const enabledWidgetIds = isEnabled
          ? region.enabledWidgetIds.filter((widgetId) => widgetId !== action.widgetId)
          : [...region.enabledWidgetIds, action.widgetId];
        const fallbackWidgetId = getNextEnabledWidgetId(region, action.widgetId);

        return {
          ...region,
          activeWidgetId: isEnabled && fallbackWidgetId ? fallbackWidgetId : action.widgetId,
          enabledWidgetIds,
          isCollapsed: action.region === 'center' ? false : enabledWidgetIds.length === 0 ? true : region.isCollapsed,
        };
      });
    }
    case 'setRegionWidgetCollapsed': {
      if (action.region === 'center') {
        return state;
      }

      return updateActiveWidgetRegion(state, action.region, (region) => ({
        ...region,
        isCollapsed: action.isCollapsed,
      }));
    }
    case 'setRegionWidgetSize': {
      return updateActiveWidgetRegion(state, action.region, (region) => ({
        ...region,
        sizePx: clampPanelSize(action.region, action.sizePx),
      }));
    }
    case 'submitInvocationSnapshot': {
      return updateActiveProject(state, (project) =>
        submitInvocationSnapshot(project, action.backendSupportsCancellation)
      );
    }
    case 'submitResolvedInvocationSnapshot': {
      return updateActiveProject(state, (project) =>
        submitInvocationSnapshot(
          project,
          action.backendSupportsCancellation,
          resolveInvocationRoute(project, 'global', action.route)
        )
      );
    }
    case 'cancelQueueItem': {
      return updateActiveProject(state, (project) => ({
        ...project,
        queue: {
          items: project.queue.items.map((item) => {
            if (item.id !== action.queueItemId || !item.cancellable || item.status !== 'pending') {
              return item;
            }

            return { ...item, status: 'cancelled' };
          }),
        },
      }));
    }
    case 'undoProjectChange': {
      return updateActiveProject(state, (project) => {
        const undoEntry = project.undoRedo.past.at(-1);

        if (!undoEntry) {
          return project;
        }

        const restoredProject = restoreUndoSnapshot(project, undoEntry.project);

        return {
          ...restoredProject,
          events: project.events,
          graphHistory: project.graphHistory,
          queue: project.queue,
          undoRedo: {
            future: [
              {
                createdAt: now(),
                id: createId('redo'),
                label: undoEntry.label,
                project: createUndoSnapshot(project),
              },
              ...project.undoRedo.future,
            ].slice(0, HISTORY_LIMIT),
            past: project.undoRedo.past.slice(0, -1),
          },
        };
      });
    }
    case 'redoProjectChange': {
      return updateActiveProject(state, (project) => {
        const redoEntry = project.undoRedo.future[0];

        if (!redoEntry) {
          return project;
        }

        const restoredProject = restoreUndoSnapshot(project, redoEntry.project);

        return {
          ...restoredProject,
          events: project.events,
          graphHistory: project.graphHistory,
          queue: project.queue,
          undoRedo: {
            future: project.undoRedo.future.slice(1),
            past: [
              ...project.undoRedo.past,
              {
                createdAt: now(),
                id: createId('undo'),
                label: redoEntry.label,
                project: createUndoSnapshot(project),
              },
            ].slice(-HISTORY_LIMIT),
          },
        };
      });
    }
    case 'hydrateWorkbench': {
      return normalizeWorkbenchState(action.state);
    }
    case 'autosaveStarted': {
      return { ...state, autosave: { status: 'saving' } };
    }
    case 'autosaveSucceeded': {
      return { ...state, autosave: { lastSavedAt: action.savedAt, status: 'saved' } };
    }
    case 'autosaveFailed': {
      return { ...state, autosave: { error: action.error, status: 'error' } };
    }
    case 'recordWidgetFailure': {
      const hasFailure = state.widgetFailures.some((failure) => failure.widgetId === action.failure.widgetId);

      if (hasFailure) {
        return state;
      }

      return {
        ...state,
        errorLog: [action.failure.details, ...state.errorLog].slice(0, ERROR_LOG_LIMIT),
        widgetFailures: [action.failure, ...state.widgetFailures],
      };
    }
    case 'recordError': {
      return { ...state, errorLog: [action.message, ...state.errorLog].slice(0, ERROR_LOG_LIMIT) };
    }
  }
};

export type { WorkbenchAction };
