import type { ProjectGraphAction } from '@features/workflow/core/document';
import type { ForLoopValidationReason } from '@features/workflow/core/forLoops';
import type { ProjectGraphState, ProjectWorkflowSource, XYPosition } from '@features/workflow/core/types';
import type { LogSource } from '@platform/logging/contracts';

export type WorkflowRegion = 'left' | 'right' | 'bottom' | 'center' | 'dialog' | 'popover' | 'floating';

/**
 * Mirror Workbench regions structurally; app/WorkflowUiAdapter widgets wiring catches drift without a forbidden
 * feature-to-workbench import.
 */
export type WorkflowWidgetPanelRegion = 'left' | 'right' | 'bottom' | 'center';

/**
 * Mirror invocation source IDs structurally; adapter and WidgetActionsMenu wiring provide type checks across the
 * ownership boundary.
 */
export type WorkflowInvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'video' | 'canvas';

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

/** Where an edit or history step lands; absent means the active project's active workflow. */
export interface WorkflowTarget {
  projectId: string;
  workflowId: string;
}

export interface WorkflowCommands {
  /** Adds and activates a document in the active project; returns the id the project knows it by. */
  addWorkflow(
    document: ProjectGraphState,
    options: { label: string; reusePlaceholder?: boolean; source?: ProjectWorkflowSource }
  ): string;
  createWorkflow(): string;
  duplicateWorkflow(workflowId: string, copyName: string): string | null;
  editGraph(action: ProjectGraphAction, target?: WorkflowTarget): void;
  redo(target?: WorkflowTarget): void;
  removeWorkflow(workflowId: string): void;
  renameWorkflow(workflowId: string, name: string): void;
  selectWorkflow(workflowId: string): void;
  /** Records a publication target; ignored once that project or workflow no longer exists. */
  setWorkflowSource(target: WorkflowTarget, source: ProjectWorkflowSource | undefined): void;
  undo(target?: WorkflowTarget): void;
}

export interface WorkflowWidgetCommands {
  open(options: { region: WorkflowWidgetPanelRegion; widgetId: string }): void;
  patchValues(widgetId: string, values: Record<string, unknown>): void;
}

export interface WorkflowNodeExecutionState {
  status: 'running' | 'completed' | 'failed';
  progress: number | null;
  progressMessage: string | null;
  outputImageUrl: string | null;
  /** The most recent invocation result of the current run. */
  latestOutput: unknown;
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
    type?: 'default' | 'loop_linkage';
  }>;
  backendGraph?: unknown;
}

export interface GraphPreviewNotice {
  id: string;
  message: string;
  nodeId?: string;
}

export interface GraphPreviewSummaryRow {
  id: string;
  label: string;
  value: string;
}

export interface GraphPreviewProvenance {
  label: string;
}

/**
 * isLive distinguishes recompilation from current project state from replaying a saved compiled graph and controls
 * the dialog subtitle.
 */
export interface GraphPreviewSourceState {
  graph: WorkflowPreviewGraph | null;
  /** Human-readable result destination (e.g. "Gallery"), set for every source by the builder. */
  destinationLabel: string | null;
  invalidReasons: Array<string | ForLoopValidationReason>;
  isLive: boolean;
  notices: GraphPreviewNotice[];
  positionHints?: Record<string, XYPosition>;
  /** nodeId → field → display text, for fields whose real value the preview can't show verbatim (e.g. a randomized seed). */
  resolvedInputOverrides?: Record<string, Record<string, string>>;
  summaryRows: GraphPreviewSummaryRow[];
  getProvenance?: (nodeId: string, fieldName: string) => GraphPreviewProvenance | null;
}

/** Attribution for editor timings. */
export type WorkflowPerfSource = LogSource;
