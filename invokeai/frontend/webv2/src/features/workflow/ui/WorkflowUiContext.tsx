import type { ProjectGraphState } from '@features/workflow/core/types';
import type { ReactNode } from 'react';

import { useExternalStoreSelector, type EqualityFn } from '@platform/state/selectors';
import { createContext, use, useCallback, useSyncExternalStore } from 'react';

import type {
  WorkflowCommands,
  WorkflowGraphHistoryEntry,
  WorkflowInvocationSourceId,
  WorkflowNodeExecutionState,
  WorkflowPerfSource,
  WorkflowWidgetCommands,
} from './contracts';

export interface WorkflowPreferences {
  reduceMotion: boolean;
  themeId: 'classic' | 'light' | 'forest' | 'mono' | 'ultradark';
  workflowEdgeStyle: 'curved' | 'square';
  workflowShowMinimap: boolean;
  workflowSnapToGrid: boolean;
  workflowValidateConnections: boolean;
}

export interface WorkflowProjectSnapshot {
  galleryValues: Record<string, unknown>;
  graphHistory: readonly WorkflowGraphHistoryEntry[];
  id: string;
  isWorkflowRunning: boolean;
  projectGraph: ProjectGraphState;
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
  ): { canInvoke: boolean; label: string; validationMessage?: string } | null;
  invoke(sourceId?: WorkflowInvocationSourceId): Promise<boolean>;
}

/**
 * Workflow's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface WorkflowUiAdapter {
  capabilities: WorkflowReadPort<WorkflowCapabilities>;
  preferences: WorkflowReadPort<WorkflowPreferences>;
  project: WorkflowReadPort<WorkflowProjectSnapshot>;
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
  registerModalHotkeyLayer(id: string): () => void;
  nodeExecution: {
    get(nodeId: string): WorkflowNodeExecutionState | null;
    subscribe(nodeId: string, listener: () => void): () => void;
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

export const useWorkflowNodeExecutionState = (nodeId: string): WorkflowNodeExecutionState | null => {
  const { nodeExecution } = useWorkflowUi();
  const subscribe = useCallback(
    (listener: () => void) => nodeExecution.subscribe(nodeId, listener),
    [nodeExecution, nodeId]
  );
  const getSnapshot = useCallback(() => nodeExecution.get(nodeId), [nodeExecution, nodeId]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};
