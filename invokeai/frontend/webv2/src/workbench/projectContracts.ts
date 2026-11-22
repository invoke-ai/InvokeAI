import type { BackendConnectionStatus } from '@platform/transport/types';

import type { CanvasStateContractV3 } from './canvas-engine/api';
import type { CanvasLoadRefusal } from './canvasLoadContracts';
import type { GraphContract } from './graphContracts';
import type { InvocationControllerState } from './invocationContracts';
import type {
  FloatingWidgetGeometry,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverrides,
  LayoutPresetOverrides,
  LayoutPresetRouteOverrides,
} from './layoutContracts';
import type { ProjectEvent } from './projectEventContracts';
import type { ProjectWorkflowCollection, ProjectWorkflowHistories } from './projectWorkflows';
import type { WorkbenchQueueState } from './queueHistoryContracts';
import type { ProjectSettings, ScopedDefaults } from './settings/contracts';
import type { WidgetFailure, WidgetInstanceContract, WidgetInstanceId, WidgetTypeId } from './widgetContracts';
import type { WorkspaceRecord } from './workspace/accountWorkspaces';
import type { StoredProjectLayout } from './workspace/migrateProjectLayout';
import type { CustomizeSessions } from './workspace/structureLock';
import type { WorkspaceViews } from './workspace/view';

export type { ProjectEvent, ProjectEventType } from './projectEventContracts';

export interface Project {
  id: string;
  name: string;
  settings: ProjectSettings;
  /** The workspace the project is on; what it shows is that workspace's structure seen through `views`. */
  workspaceId: LayoutPresetId;
  /** How the project looks at each workspace: sparse overrides of its structure's defaults. */
  views: WorkspaceViews;
  invocation: InvocationControllerState;
  workflows: ProjectWorkflowCollection;
  /** Session-only graph edit histories by workflow id; never persisted. */
  workflowHistories: ProjectWorkflowHistories;
  /** Widget values by instance id; an instance a structure drops keeps its values for when it is placed again. */
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  /** Convenience, not layout: undo leaves it out. */
  lastFloatingGeometry?: Record<WidgetInstanceId, FloatingWidgetGeometry>;
  /**
   * The arrangement a document stored before structure and view split, as loaded: set only between loading such a
   * document and the reducer step that migrates it against the account, and written back unchanged until then.
   */
  legacyLayout?: StoredProjectLayout;
  /**
   * The stored layout a project migrated during its account's adoption is written back with, unchanged, until the
   * account that adopted is saved: a load that never saved it migrates and adopts again. Transient.
   */
  heldLegacyLayout?: StoredProjectLayout;
  /** Workspaces whose own arrangement this project lost when its stored layout migrated: told once, then cleared. */
  droppedArrangements?: readonly LayoutPresetId[];
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
  /** Projects active at least once in this editor session; a first activation enters its workspace. Transient. */
  enteredProjectIds?: readonly string[];
  /** Projects in Customize on their workspace. Transient: never persisted, synced across tabs or undone. */
  customizing?: CustomizeSessions;
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

/** `workspace-offer`: an open that found nothing to show, or a locked structural gesture; a repeat coalesces. */
export type WorkbenchNotificationCategory = 'enqueue' | 'run-outcome' | 'workspace-offer';

/** Plain data, never a closure: notifications live in workbench state. */
export type WorkbenchNotificationAction =
  | { kind: 'open-widget'; widgetId: WidgetTypeId }
  /** Adds the widget to the current workspace, on the surface the open asked for: the structural add an offer makes. */
  | { kind: 'add-widget'; widgetId: WidgetTypeId; surface?: 'stage' | 'panel' }
  /** Enters Customize on the active project's workspace: what a locked structural gesture offers. */
  /** Customize, on the named workspace (entering it first) or else on the one the project is on. */
  | { kind: 'customize-workspace'; workspaceId?: LayoutPresetId }
  /** The one-time offer to existing accounts: turns "Rearrange without Customize" off, or keeps it. */
  | { kind: 'lock-workspaces' }
  | { kind: 'keep-rearranging' }
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
  /** Values that are lists of names, each translated by its key (else shown as its text) when the message is shown. */
  messageListValues?: Readonly<Record<string, readonly { key?: string; text: string }[]>>;
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
  | { kind: 'switchPreset'; presetId: LayoutPresetId }
  | { kind: 'removeWidget'; instanceId: WidgetInstanceId }
  | { kind: 'resetView'; presetId: LayoutPresetId }
  | { kind: 'updateWorkspace'; presetId: LayoutPresetId };

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
  notice?: LayoutUndoNotice;
  mergeKey?: string;
  mergedAt?: string;
  /**
   * A step that changed a workspace in the account (a structural gesture outside Customize, or Done): what the account
   * held for it before and after. Undoing puts `before` back, for every project on that workspace, only while the
   * account still holds `after`; redo the other way round.
   */
  workspace?: { workspaceId: LayoutPresetId; before: WorkspaceRecord; after: WorkspaceRecord };
}

/** Project undo preserves the live canvas and workflow documents; the engine and workflow histories own those. */
export interface ProjectUndoSnapshot {
  workspaceId: LayoutPresetId;
  views: WorkspaceViews;
  invocation: InvocationControllerState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
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
  /**
   * Set once the account's workspaces took over the arrangements projects stored before structure and view split
   * (the most recently used project's became its workspaces). A fresh account starts with it.
   */
  workspaceLayoutsAdopted?: true;
}
