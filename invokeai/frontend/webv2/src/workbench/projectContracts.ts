import type { BackendConnectionStatus } from '@platform/transport/types';

import type { CanvasStateContractV3 } from './canvas-engine/api';
import type { CanvasLoadRefusal } from './canvasLoadContracts';
import type { GraphContract } from './graphContracts';
import type { InvocationControllerState } from './invocationContracts';
import type {
  FloatingWidgetState,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverrides,
  LayoutPresetOverrides,
  LayoutPresetRouteOverrides,
  ProjectLayoutState,
  WidgetRegion,
  WidgetRegionState,
} from './layoutContracts';
import type { ProjectEvent } from './projectEventContracts';
import type { ProjectWorkflowCollection, ProjectWorkflowHistories } from './projectWorkflows';
import type { WorkbenchQueueState } from './queueHistoryContracts';
import type { ProjectSettings } from './settings/contracts';
import type { WidgetFailure, WidgetInstanceContract, WidgetInstanceId, WidgetTypeId } from './widgetContracts';

export type { ProjectEvent, ProjectEventType } from './projectEventContracts';

export interface Project {
  id: string;
  name: string;
  settings: ProjectSettings;
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  /** The project's workflows and which one is active; the active document compiles to a `GraphContract` at invoke time. */
  workflows: ProjectWorkflowCollection;
  /** Session-only graph edit histories by workflow id; never persisted. */
  workflowHistories: ProjectWorkflowHistories;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  /** Floating instances leave their region's instanceIds; absent in older projects means no floating windows. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
  canvas: CanvasStateContractV3;
  promptHistory: PromptHistoryItem[];
  undoRedo: UndoRedoHistory;
  queue: WorkbenchQueueState;
  events: ProjectEvent[];
}

/**
 * A persisted project document that was not loaded: written by a newer version, or structurally damaged. `raw` is
 * the untouched document, kept for recovery; nothing is written back over it.
 */
export type ProjectDocumentLoadRefusal =
  | { raw: unknown; scope: 'project-document'; status: 'unsupported-version'; version: number }
  | { raw: unknown; scope: 'project-document'; status: 'malformed'; reason: string };

interface RefusedWorkbenchProjectBase {
  projectId: string;
  projectName: string;
  raw: unknown;
}

export type RefusedWorkbenchProject = RefusedWorkbenchProjectBase &
  (
    | { refusal: CanvasLoadRefusal; source: 'canvas'; queueItem?: never }
    | { refusal: ProjectDocumentLoadRefusal; source: 'project-document'; queueItem?: never }
  );

export type ProjectLoadResult =
  | { status: 'loaded'; project: Project }
  | { status: 'refused'; refused: RefusedWorkbenchProject }
  | { status: 'unavailable' };

export interface WorkbenchState {
  projects: Project[];
  activeProjectId: string;
  backendConnection: BackendConnectionState;
  notifications: WorkbenchNotification[];
  autosave: AutosaveState;
  account: AccountState;
  widgetFailures: WidgetFailure[];
}

export interface BackendConnectionState {
  status: BackendConnectionStatus;
  error?: string;
  lastConnectedAt?: string;
  lastDisconnectedAt?: string;
}

export type WorkbenchNotificationKind = 'error' | 'success' | 'info';

/** Machine categories for toast policy. Absent category = always toast. */
/** `run-outcome`: a queue item the user submitted failed or was cancelled. */
export type WorkbenchNotificationCategory = 'enqueue' | 'run-outcome';

export interface WorkbenchNotification {
  id: string;
  kind: WorkbenchNotificationKind;
  title: string;
  titleKey?: string;
  message?: string;
  messageKey?: string;
  createdAt: string;
  projectId?: string;
  isRead: boolean;
  /** Machine category for toast policy; absent = always toast. */
  category?: WorkbenchNotificationCategory;
  /** Coalesced repeat count (see addNotification); absent = 1. */
  occurrenceCount?: number;
}

export interface PromptHistoryItem {
  positivePrompt: string;
  negativePrompt: string | null;
}

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
  /** Edits that arrive as a stream (typing, dragging) share a key so they fold into one step. */
  mergeKey?: string;
  /** When the entry last absorbed a same-key edit; the merge window runs from here. */
  mergedAt?: string;
}

/** Project undo preserves the live canvas and workflow documents; the engine and workflow histories own those. */
export interface ProjectUndoSnapshot {
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  /** Captured with widgetRegions: regions and floating windows are one placement fact. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
}

export interface UndoRedoHistory {
  past: UndoRedoEntry[];
  future: UndoRedoEntry[];
}

export interface AutosaveState {
  /** `pending`: persisted content changed since the last acknowledged save and its save has not started yet. */
  status: 'idle' | 'pending' | 'saving' | 'saved' | 'error';
  lastSavedAt?: string;
  error?: string;
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
  customLayoutPresets?: LayoutPreset[];
  /** One account-wide order shared by every layout-preset surface. */
  layoutPresetOrder?: LayoutPresetId[];
  /** Saved name and icon edits for built-in presets. */
  layoutPresetMetadataOverrides?: LayoutPresetMetadataOverrides;
  /** Saved edits to a built-in preset's arrangement; see {@link LayoutPresetOverrides}. */
  layoutPresetOverrides?: LayoutPresetOverrides;
  /** Saved edits to built-in preset routes, kept separate from spatial layout drift. */
  layoutPresetRouteOverrides?: LayoutPresetRouteOverrides;
}
