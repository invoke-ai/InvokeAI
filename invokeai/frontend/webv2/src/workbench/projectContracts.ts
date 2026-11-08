import type { BackendConnectionStatus } from '@platform/transport/types';

import type { CanvasStateContractV3 } from './canvas-engine/api';
import type { CanvasLoadRefusal } from './canvasLoadContracts';
import type { GraphContract } from './graphContracts';
import type { InvocationControllerState } from './invocationContracts';
import type {
  FloatingWidgetGeometry,
  FloatingWidgetState,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverrides,
  LayoutPresetOverrides,
  LayoutPresetRouteOverrides,
  LayoutPresetWorkingCopy,
  ProjectLayoutState,
  WidgetRegion,
  WidgetRegionState,
} from './layoutContracts';
import type { ProjectEvent } from './projectEventContracts';
import type { ProjectWorkflowCollection, ProjectWorkflowHistories } from './projectWorkflows';
import type { WorkbenchQueueState } from './queueHistoryContracts';
import type { ProjectSettings, ScopedDefaults } from './settings/contracts';
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
  /**
   * The last window geometry of instances that are not floating now. It is a convenience, not layout: presets,
   * layout drift, and undo leave it out. Absent means nothing is remembered.
   */
  lastFloatingGeometry?: Record<WidgetInstanceId, FloatingWidgetGeometry>;
  /**
   * This project's arrangement of each preset it switched away from while that arrangement differed from the saved
   * one, oldest first, at most one per preset. Arrangement only (no widget state), and never the active preset: the
   * live layout is its working copy. Absent means every inactive preset opens as saved.
   */
  presetWorkingLayouts?: LayoutPresetWorkingCopy[];
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
  /** Mirror of the user's `WorkbenchPreferences.scopedDefaults`: reads resolve pins over defaults in one snapshot. */
  scopedDefaults: ScopedDefaults;
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

/**
 * A follow-up a notification offers, as plain data: notifications live in workbench state, so they never hold a
 * closure. The notification toaster resolves each kind to a label and a handler (see
 * `shell/notifications/README.md`); a kind it cannot offer right now is left out.
 */
export type WorkbenchNotificationAction =
  | { kind: 'open-widget'; widgetId: WidgetTypeId }
  /** Opens the queue at this backend item. */
  | { kind: 'reveal-queue-item'; itemId: number }
  /**
   * Re-runs these failed or canceled backend items of one run from their saved sessions, once, in one request.
   * `canvasProjectId`: a Canvas run's project, whose rerun is staged only by a tab with it open.
   */
  | { kind: 'retry-queue-item'; itemIds: readonly number[]; canvasProjectId?: string }
  /**
   * Runs the closure `useNotify().undoable` holds in its session-only registry, at most once. With `projectId`, only
   * while that project is open. `label` replaces "Undo" when the button brings back something other than the
   * reported action.
   */
  | { kind: 'undo'; undoId: string; projectId?: string; label?: string };

export interface WorkbenchNotification {
  id: string;
  kind: WorkbenchNotificationKind;
  title: string;
  titleKey?: string;
  /** Interpolation values for `titleKey`; `title` holds the same text in English. */
  titleValues?: Readonly<Record<string, string | number>>;
  message?: string;
  messageKey?: string;
  messageValues?: Readonly<Record<string, string | number>>;
  createdAt: string;
  projectId?: string;
  isRead: boolean;
  /** Machine category for toast policy; absent = always toast. */
  category?: WorkbenchNotificationCategory;
  /** Coalesced repeat count (see addNotification); absent = 1. */
  occurrenceCount?: number;
  /** Buttons its toast shows, primary first. */
  actions?: readonly WorkbenchNotificationAction[];
  /** Its toast stays until dismissed, for failures the user must not miss. */
  persistent?: boolean;
}

export interface PromptHistoryItem {
  positivePrompt: string;
  negativePrompt: string | null;
}

/**
 * A layout change worth reporting with an Undo toast; the step that made it carries this (see
 * `shell/LayoutUndoNotices.tsx`). Steps without one, such as a resize or a route change, are undone only by the
 * layout undo command.
 */
export type LayoutUndoNotice =
  /** The active preset's saved arrangement was laid out again over unsaved changes. */
  | { kind: 'revert'; presetId: LayoutPresetId }
  /** Another preset's unsaved arrangement in this project was dropped. */
  | { kind: 'discardCopy'; presetId: LayoutPresetId }
  | { kind: 'switchPreset'; presetId: LayoutPresetId }
  | { kind: 'removeWidget'; instanceId: WidgetInstanceId };

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
  notice?: LayoutUndoNotice;
  /** Edits that arrive as a stream (a dragged divider, a moved window) share a key so they fold into one step. */
  mergeKey?: string;
  /** When the step last absorbed a same-key edit; the merge window runs from here. */
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
  /** Captured with `layout.presetId`, so undoing a preset switch brings back the working copies it moved. */
  presetWorkingLayouts?: Project['presetWorkingLayouts'];
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
}

export interface UndoRedoHistory {
  past: UndoRedoEntry[];
  future: UndoRedoEntry[];
}

export interface AutosaveState {
  /**
   * `pending`: persisted content changed since the last acknowledged save and its save has not started yet, or a save
   * that could not finish will retry. `attention`: the last save finished for every other project, but an open project
   * awaits a user decision (a sync conflict or an unsupported schema); the project sync snapshot says which.
   */
  status: 'idle' | 'pending' | 'saving' | 'saved' | 'attention' | 'error';
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
