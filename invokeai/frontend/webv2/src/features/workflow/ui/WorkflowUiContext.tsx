import type { GalleryItemRef } from '@features/gallery/contracts';
import type { ForLoopValidationReason } from '@features/workflow/core/forLoops';
import type { ProjectGraphState, ProjectWorkflowEntry, ProjectWorkflowSource } from '@features/workflow/core/types';
import type { WorkbenchThemeId } from '@theme/themes';
import type { ReactNode } from 'react';

import { useExternalStoreSelector, type EqualityFn } from '@platform/state/selectors';
import { createContext, use, useCallback, useSyncExternalStore } from 'react';

import type {
  WorkflowCommands,
  WorkflowInvocationSourceId,
  WorkflowNodeExecutionState,
  WorkflowPerfSource,
  WorkflowWidgetCommands,
} from './contracts';

export interface WorkflowPreferences {
  reduceMotion: boolean;
  themeId: WorkbenchThemeId;
  workflowEdgeStyle: 'curved' | 'square';
  workflowEdgesBehindNodes: boolean;
  workflowGroupNodesByCategory: boolean;
  workflowShowMinimap: boolean;
  workflowSnapToGrid: boolean;
  workflowValidateConnections: boolean;
}

/** How the project's own persistence stands; the editor shows it beside the library actions. */
export interface WorkflowProjectPersistence {
  status: 'pending' | 'saving' | 'saved' | 'error' | 'conflict';
  /** False when browser recovery storage is unavailable, so a pending save has no local safety net. */
  hasLocalRecovery: boolean;
  lastSavedAt: string | null;
  error: string | null;
}

export interface WorkflowProjectSnapshot {
  galleryValues: Record<string, unknown>;
  id: string;
  isWorkflowRunning: boolean;
  /** The active workflow's editable document. */
  projectGraph: ProjectGraphState;
  /** The active workflow's entry: its document plus source and run metadata. */
  activeWorkflow: ProjectWorkflowEntry;
  activeWorkflowId: string;
  /** Every workflow the project owns, in collection order; entries keep identity until edited. */
  workflows: readonly ProjectWorkflowEntry[];
  workflowValues: Record<string, unknown>;
}

export interface WorkflowCapabilities {
  canUseCache: boolean;
}

export interface WorkflowReadPort<Snapshot> {
  getSnapshot(): Snapshot;
  subscribe(listener: () => void): () => void;
}

export interface WorkflowGraphPreviewPort {
  getRoute(
    sourceId?: WorkflowInvocationSourceId
  ): { canInvoke: boolean; label: string; validationMessage?: string | ForLoopValidationReason } | null;
  invoke(sourceId?: WorkflowInvocationSourceId): Promise<boolean>;
  focusSource(sourceId?: WorkflowInvocationSourceId): void; // reveal the source's widget (provenance links)
  openWorkflowEditor(): void; // reveal the workflow editor widget
  /** Forks a document into a fresh project; a library source travels with it so the copy can update its template. */
  openDocumentInNewProject(document: ProjectGraphState, label: string, source?: ProjectWorkflowSource): void;
}

/** This UI port preserves dependency direction: Workflow cannot import Workbench. */
export interface WorkflowUiAdapter {
  capabilities: WorkflowReadPort<WorkflowCapabilities>;
  preferences: WorkflowReadPort<WorkflowPreferences>;
  project: WorkflowReadPort<WorkflowProjectSnapshot>;
  persistence: WorkflowReadPort<WorkflowProjectPersistence>;
  commands: WorkflowCommands;
  widgets: WorkflowWidgetCommands;
  getProjectGraph(): ProjectGraphState;
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    success(title: string, message?: string): void;
  };
  performance: {
    mark(name: string, source: WorkflowPerfSource): void;
    measure(name: string, start: string, source: WorkflowPerfSource, end?: string): void;
    time<T>(name: string, source: WorkflowPerfSource, callback: () => T): T;
  };
  /** Raises the gallery and selects the item, as the other media fields' find buttons do. */
  findInGallery(ref: GalleryItemRef): void;
  /** Leaves the editor for the model manager's Add Models section, searching for `query`. */
  openAddModels(query: string): void;
  registerModalHotkeyLayer(id: string): () => void;
  nodeExecution: {
    get(nodeId: string): WorkflowNodeExecutionState | null;
    subscribe(nodeId: string, listener: () => void): () => void;
    /** Which project workflow the tracked run came from; progress for another copy stays out of this editor. */
    getOrigin(): { projectId: string; workflowId: string } | null;
    subscribeOrigin(listener: () => void): () => void;
  };
}

const WorkflowUiContext = createContext<WorkflowUiAdapter | null>(null);
const WorkflowGraphPreviewContext = createContext<WorkflowGraphPreviewPort | null>(null);

export const WorkflowUiProvider = ({ adapter, children }: { adapter: WorkflowUiAdapter; children: ReactNode }) => (
  <WorkflowUiContext value={adapter}>{children}</WorkflowUiContext>
);

export const WorkflowGraphPreviewProvider = ({
  adapter,
  children,
}: {
  adapter: WorkflowGraphPreviewPort;
  children: ReactNode;
}) => <WorkflowGraphPreviewContext value={adapter}>{children}</WorkflowGraphPreviewContext>;

export const useWorkflowUi = (): WorkflowUiAdapter => {
  const adapter = use(WorkflowUiContext);

  if (!adapter) {
    throw new Error('Workflow UI requires an App-composed WorkflowUiProvider.');
  }

  return adapter;
};

export const useWorkflowProjectSelector = <Selected,>(
  selector: (project: WorkflowProjectSnapshot) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => {
  const { project } = useWorkflowUi();
  return useExternalStoreSelector(project.subscribe, project.getSnapshot, selector, isEqual);
};

export const useWorkflowHostCommands = () => {
  const ui = useWorkflowUi();
  return { widgets: ui.widgets, workflows: ui.commands };
};

export const useWorkflowPreferencesSelector = <Selected,>(
  selector: (preferences: WorkflowPreferences) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => {
  const { preferences } = useWorkflowUi();
  return useExternalStoreSelector(preferences.subscribe, preferences.getSnapshot, selector, isEqual);
};

export const useWorkflowCapabilitiesSelector = <Selected,>(
  selector: (capabilities: WorkflowCapabilities) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => {
  const { capabilities } = useWorkflowUi();
  return useExternalStoreSelector(capabilities.subscribe, capabilities.getSnapshot, selector, isEqual);
};

export const useWorkflowGraphPreview = (): WorkflowGraphPreviewPort => {
  const graphPreview = use(WorkflowGraphPreviewContext);
  if (!graphPreview) {
    throw new Error('Workflow graph preview requires an App-composed WorkflowGraphPreviewProvider.');
  }
  return graphPreview;
};

export const useWorkflowNotifications = () => useWorkflowUi().notifications;

export const useOpenAddModels = (): ((query: string) => void) => useWorkflowUi().openAddModels;

export const useWorkflowPersistenceSelector = <Selected,>(
  selector: (persistence: WorkflowProjectPersistence) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => {
  const { persistence } = useWorkflowUi();
  return useExternalStoreSelector(persistence.subscribe, persistence.getSnapshot, selector, isEqual);
};

/** A node's execution state, only while the tracked run originated from the workflow this editor shows. */
export const useWorkflowNodeExecutionState = (nodeId: string): WorkflowNodeExecutionState | null => {
  const { nodeExecution, project } = useWorkflowUi();
  const subscribe = useCallback(
    (listener: () => void) => {
      const unsubscribers = [
        nodeExecution.subscribe(nodeId, listener),
        nodeExecution.subscribeOrigin(listener),
        project.subscribe(listener),
      ];
      return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
    },
    [nodeExecution, nodeId, project]
  );
  const getSnapshot = useCallback(() => {
    const origin = nodeExecution.getOrigin();
    const snapshot = project.getSnapshot();
    return origin && origin.projectId === snapshot.id && origin.workflowId === snapshot.activeWorkflowId
      ? nodeExecution.get(nodeId)
      : null;
  }, [nodeExecution, nodeId, project]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};
