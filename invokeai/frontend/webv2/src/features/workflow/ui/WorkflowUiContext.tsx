import type { ProjectGraphState } from '@features/workflow/core/types';
import type { ComponentType, ReactNode } from 'react';

import { createContext, use, useCallback, useSyncExternalStore } from 'react';

import type {
  WorkflowCommands,
  WorkflowGraphHistoryEntry,
  WorkflowInvocationSourceId,
  WorkflowModelSelectProps,
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

/**
 * Workflow's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface WorkflowUiAdapter {
  ModelSelect: ComponentType<WorkflowModelSelectProps>;
  activeProjectId: string;
  canUseCache: boolean;
  galleryValues: Record<string, unknown>;
  graphHistory: readonly WorkflowGraphHistoryEntry[];
  isWorkflowRunning: boolean;
  preferences: WorkflowPreferences;
  projectGraph: ProjectGraphState;
  workflowValues: Record<string, unknown>;
  commands: WorkflowCommands;
  widgets: WorkflowWidgetCommands;
  getProjectGraph(): ProjectGraphState;
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    success(title: string, message?: string): void;
  };
  graphPreview: {
    getRoute(
      sourceId?: WorkflowInvocationSourceId
    ): { canInvoke: boolean; label: string; validationMessage?: string } | null;
    invoke(sourceId?: WorkflowInvocationSourceId): Promise<boolean>;
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

export const WorkflowUiProvider = ({ adapter, children }: { adapter: WorkflowUiAdapter; children: ReactNode }) => (
  <WorkflowUiContext value={adapter}>{children}</WorkflowUiContext>
);

export const useWorkflowUi = (): WorkflowUiAdapter => {
  const adapter = use(WorkflowUiContext);

  if (!adapter) {
    throw new Error('Workflow UI requires an App-composed WorkflowUiProvider.');
  }

  return adapter;
};

export const useWorkflowProjectSelector = <Selected,>(
  selector: (project: {
    id: string;
    projectGraph: ProjectGraphState;
    graphHistory: readonly WorkflowGraphHistoryEntry[];
    galleryValues: Record<string, unknown>;
    workflowValues: Record<string, unknown>;
    isWorkflowRunning: boolean;
  }) => Selected
): Selected => {
  const ui = useWorkflowUi();
  return selector({
    galleryValues: ui.galleryValues,
    graphHistory: ui.graphHistory,
    id: ui.activeProjectId,
    isWorkflowRunning: ui.isWorkflowRunning,
    projectGraph: ui.projectGraph,
    workflowValues: ui.workflowValues,
  });
};

export const useWorkflowHostCommands = () => {
  const ui = useWorkflowUi();
  return { widgets: ui.widgets, workflows: ui.commands };
};

export const useWorkflowPreferencesSelector = <Selected,>(
  selector: (preferences: WorkflowPreferences) => Selected,
  _isEqual?: (left: Selected, right: Selected) => boolean
): Selected => selector(useWorkflowUi().preferences);

export const useWorkflowNotifications = () => useWorkflowUi().notifications;

export const WorkflowModelSelect = (props: WorkflowModelSelectProps) => {
  const { ModelSelect } = useWorkflowUi();
  return <ModelSelect {...props} />;
};

export const useWorkflowNodeExecutionState = (nodeId: string): WorkflowNodeExecutionState | null => {
  const { nodeExecution } = useWorkflowUi();
  const subscribe = useCallback(
    (listener: () => void) => nodeExecution.subscribe(nodeId, listener),
    [nodeExecution, nodeId]
  );
  const getSnapshot = useCallback(() => nodeExecution.get(nodeId), [nodeExecution, nodeId]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};
