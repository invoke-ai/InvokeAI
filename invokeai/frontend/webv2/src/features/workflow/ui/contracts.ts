import type { ProjectGraphAction } from '@features/workflow/core/document';
import type { ProjectGraphState } from '@features/workflow/core/types';

export type WorkflowRegion = 'left' | 'right' | 'bottom' | 'center' | 'dialog' | 'popover';

/**
 * Panel regions workflow surfaces may open/select widgets in. Structural
 * mirror of workbench's `WidgetRegion` (`workbench/layoutContracts.ts`) —
 * the feature may not import workbench, so drift is caught as a tsc error at
 * the check site: the `widgets` wiring in `app/WorkflowUiAdapter.tsx`.
 */
export type WorkflowWidgetPanelRegion = 'left' | 'right' | 'bottom' | 'center';

/**
 * Invocation sources the graph preview can route. Structural mirror of
 * workbench's `InvocationSourceId` (`workbench/invocationContracts.ts`) —
 * drift is caught as a tsc error at the check sites: the `graphPreview`
 * wiring in `app/WorkflowUiAdapter.tsx` and the `GraphPreviewDialog` mount in
 * `workbench/widget-frame/WidgetActionsMenu.tsx`.
 */
export type WorkflowInvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'canvas';

export interface WorkflowRuntimeApi {
  instanceId: string;
  typeId: string;
  region: WorkflowRegion;
  commands: {
    register(command: { id: string; title: string; handler: () => unknown }): () => void;
  };
  hotkeys: {
    register(hotkey: { id: string; commandId: string; defaultKeys: string[]; title: string }): () => void;
  };
}

export interface WorkflowWidgetViewProps {
  region: WorkflowRegion;
  runtime: WorkflowRuntimeApi;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export interface WorkflowWidgetLabelProps {
  region: WorkflowRegion;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export interface WorkflowGraphHistoryEntry {
  id: string;
  createdAt: string;
  label: string;
  document?: ProjectGraphState;
}

export interface WorkflowCommands {
  bindLibraryWorkflow(libraryWorkflowId: string): void;
  editGraph(action: ProjectGraphAction): void;
  replace(document: ProjectGraphState, label: string): void;
  redo(): void;
  restoreSnapshot(snapshotId: string): void;
  saveSnapshot(): void;
  undo(): void;
}

export interface WorkflowWidgetCommands {
  open(options: { region: WorkflowWidgetPanelRegion; widgetId: string }): void;
  patchValues(widgetId: string, values: Record<string, unknown>): void;
  select(options: { region: WorkflowWidgetPanelRegion; widgetId: string }): void;
}

export interface WorkflowModel {
  base: string;
  hash?: string;
  key: string;
  name: string;
  type: string;
}

export interface WorkflowModelSelectProps {
  className?: string;
  filter?: (model: WorkflowModel) => boolean;
  id?: string;
  invalid?: boolean;
  isClearable?: boolean;
  modelTypes: string[];
  onChange(model: WorkflowModel | null): void;
  size?: 'xs' | 'sm' | 'md';
  value: string | null;
}

export interface WorkflowNodeExecutionState {
  status: 'running' | 'completed' | 'failed';
  progress: number | null;
  progressMessage: string | null;
  outputImageUrl: string | null;
  error: string | null;
}

export interface WorkflowPreviewGraph {
  id: string;
  label?: string;
  updatedAt?: string;
  version?: 1;
  nodes: Array<{ id: string; type: string; inputs: Record<string, unknown> }>;
  edges: Array<{
    id: string;
    sourceNodeId: string;
    sourceField: string;
    targetNodeId: string;
    targetField: string;
  }>;
  backendGraph?: unknown;
}

export interface WorkflowPerfSource {
  instanceId: string;
  kind: 'widget';
  projectId: string;
  region: WorkflowRegion;
  typeId: string;
}
