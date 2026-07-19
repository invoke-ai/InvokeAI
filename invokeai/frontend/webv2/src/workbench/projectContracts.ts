import type { ProjectGraphState } from '@features/workflow/contracts';
import type { BackendConnectionStatus } from '@platform/transport/types';

import type { CanvasStateContractV2 } from './canvas-engine/api';
import type { GraphContract } from './graphContracts';
import type { InvocationControllerState } from './invocationContracts';
import type {
  LayoutPreset,
  LayoutPresetId,
  ProjectLayoutState,
  WidgetRegion,
  WidgetRegionState,
} from './layoutContracts';
import type { WorkbenchQueueState } from './queueHistoryContracts';
import type { ProjectSettings } from './settings/contracts';
import type { WidgetFailure, WidgetInstanceContract, WidgetInstanceId, WidgetTypeId } from './widgetContracts';

export type ProjectEventType =
  | 'project-created'
  | 'layout-updated'
  | 'invocation-updated'
  | 'queue-submitted'
  | 'canvas-layer-accepted'
  | 'graph-replaced'
  | 'graph-snapshot-saved';

export interface Project {
  id: string;
  name: string;
  /**
   * Set on recovery forks created when a save loses a revision race: the id
   * of the root project this recovered from (chains collapse to the root, so
   * a recovery of a recovery still points at the original).
   */
  recoveryOf?: string;
  recoveredAt?: string;
  settings: ProjectSettings;
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  /** The one active project graph: an editable workflow document, compiled to a `GraphContract` at invoke time. */
  projectGraph: ProjectGraphState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
  canvas: CanvasStateContractV2;
  graphHistory: GraphHistorySnapshot[];
  promptHistory: PromptHistoryItem[];
  undoRedo: UndoRedoHistory;
  queue: WorkbenchQueueState;
  events: ProjectEvent[];
}

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

export interface WorkbenchNotification {
  id: string;
  kind: WorkbenchNotificationKind;
  title: string;
  message?: string;
  createdAt: string;
  projectId?: string;
  isRead: boolean;
}

/**
 * One entry of the project's graph history. Queue submissions record the
 * compiled `graph`; workflow snapshots (manual save, pre-replacement) record
 * the editable `document`, which is what makes them restorable.
 */
export interface GraphHistorySnapshot {
  id: string;
  createdAt: string;
  label: string;
  graph?: GraphContract;
  document?: ProjectGraphState;
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
}

/**
 * Project-level undo snapshot. Deliberately excludes `canvas`: the canvas
 * rendering engine owns its own pixel-patch history, so project undo/redo
 * passes the live `project.canvas` through untouched (see `restoreUndoSnapshot`).
 */
export interface ProjectUndoSnapshot {
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  projectGraph: ProjectGraphState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
}

export interface UndoRedoHistory {
  past: UndoRedoEntry[];
  future: UndoRedoEntry[];
}

export interface ProjectEvent {
  id: string;
  type: ProjectEventType;
  createdAt: string;
  summary: string;
  runId?: string;
}

export interface AutosaveState {
  status: 'idle' | 'saving' | 'saved' | 'error';
  lastSavedAt?: string;
  error?: string;
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
  customLayoutPresets?: LayoutPreset[];
}
