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
  workflows: ProjectWorkflowCollection;
  /** Session-only graph edit histories by workflow id; never persisted. */
  workflowHistories: ProjectWorkflowHistories;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  /** Floating instances leave their region's instanceIds; absent in older projects means no floating windows. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
  /** Convenience, not layout: presets, layout drift and undo leave it out. */
  lastFloatingGeometry?: Record<WidgetInstanceId, FloatingWidgetGeometry>;
  /** Inactive presets' differing arrangements, oldest first, one per preset; never the active preset's. */
  presetWorkingLayouts?: LayoutPresetWorkingCopy[];
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
  canvas: CanvasStateContractV3;
  promptHistory: PromptHistoryItem[];
  undoRedo: UndoRedoHistory;
  queue: WorkbenchQueueState;
  events: ProjectEvent[];
}

/** `raw` is the untouched document, kept for recovery; nothing is written back over it. */
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
  scopedDefaults: ScopedDefaults;
  /** Transient: never persisted, synced across tabs or undone. */
  canvasStagingFollow?: Readonly<
    Record<string, { readonly framedSlotIds: readonly string[]; readonly pinnedSlotId: string | null }>
  >;
}

export interface BackendConnectionState {
  status: BackendConnectionStatus;
  error?: string;
  lastConnectedAt?: string;
  lastDisconnectedAt?: string;
}

export type WorkbenchNotificationKind = 'error' | 'success' | 'info';

export type WorkbenchNotificationCategory = 'enqueue' | 'run-outcome';

/** Plain data, never a closure: notifications live in workbench state. */
export type WorkbenchNotificationAction =
  | { kind: 'open-widget'; widgetId: WidgetTypeId }
  | { kind: 'reveal-queue-item'; itemId: number }
  /** `canvasProjectId`: a Canvas run's project, whose rerun is staged only by a tab with it open. */
  | { kind: 'retry-queue-item'; itemIds: readonly number[]; canvasProjectId?: string }
  /** Runs `useNotify().undoable`'s session-only closure at most once; with `projectId`, only while it is open. */
  | { kind: 'undo'; undoId: string; projectId?: string; label?: string };

export interface WorkbenchNotification {
  id: string;
  kind: WorkbenchNotificationKind;
  title: string;
  titleKey?: string;
  titleValues?: Readonly<Record<string, string | number>>;
  message?: string;
  messageKey?: string;
  messageValues?: Readonly<Record<string, string | number>>;
  createdAt: string;
  projectId?: string;
  isRead: boolean;
  /** Machine category for toast policy; absent = always toast. */
  category?: WorkbenchNotificationCategory;
  occurrenceCount?: number;
  actions?: readonly WorkbenchNotificationAction[];
  persistent?: boolean;
}

export interface PromptHistoryItem {
  positivePrompt: string;
  negativePrompt: string | null;
}

export type LayoutUndoNotice =
  | { kind: 'revert'; presetId: LayoutPresetId }
  | { kind: 'discardCopy'; presetId: LayoutPresetId }
  | { kind: 'switchPreset'; presetId: LayoutPresetId }
  | { kind: 'removeWidget'; instanceId: WidgetInstanceId };

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
  notice?: LayoutUndoNotice;
  mergeKey?: string;
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
  /** `attention`: every other project saved, but an open project awaits a user decision. */
  status: 'idle' | 'pending' | 'saving' | 'saved' | 'attention' | 'error';
  lastSavedAt?: string;
  error?: string;
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
  customLayoutPresets?: LayoutPreset[];
  layoutPresetOrder?: LayoutPresetId[];
  layoutPresetMetadataOverrides?: LayoutPresetMetadataOverrides;
  layoutPresetOverrides?: LayoutPresetOverrides;
  layoutPresetRouteOverrides?: LayoutPresetRouteOverrides;
}
