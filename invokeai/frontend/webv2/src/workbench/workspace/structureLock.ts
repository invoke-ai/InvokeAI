import type { LayoutPresetId, LayoutPresetRoute } from '@workbench/layoutContracts';

import { builtInLayoutPresetDescriptors } from '@workbench/layoutPresets';

import type { WorkspaceStructure } from './structure';
import type { WorkspaceView } from './view';

/**
 * Name, icon and route edits made in the Customize bar: Done applies them, Cancel drops them. Absent is the base's:
 * the workspace as saved, or as shipped after Reset to default. A new workspace's draft names all three.
 */
export interface CustomizeDraft {
  /** `null` is Auto: the job follows the fronted sources. */
  defaultRoute?: LayoutPresetRoute | null;
  iconId?: string;
  label?: string;
}

/** A workspace that New workspace or Save as new makes: Done creates it; until then the account has no such workspace. */
export interface NewWorkspaceDraft {
  defaultRoute: LayoutPresetRoute | null;
  iconId?: string;
  /** The id Done creates it with; the project's view of the draft is kept under it meanwhile. */
  id: LayoutPresetId;
  label: string;
}

/**
 * One project customizing the workspace it is on, or a new workspace drafted from it. Session-only: never persisted or
 * synced. Nothing reaches the account until Done; Cancel leaves the account and the project as they were.
 */
export interface CustomizeSession {
  draft: CustomizeDraft;
  /** The structure Customize edits and the screen shows. */
  structure: WorkspaceStructure;
  /** Reset to default: the draft's base is the built-in as shipped, and Done removes its overrides first. */
  resetPending?: true;
  /** Set for a new workspace's draft; the workspace at `presetId` is then left as it was. */
  newWorkspaceId?: LayoutPresetId;
  /** The account's structure when Customize began: Done writes only a change from it. */
  entryStructure: WorkspaceStructure;
  /** The project's view of the workspace when Customize began: Cancel puts it back. */
  entryView: WorkspaceView | undefined;
  /** Every step the session made (structure, view, name, icon, route, reset), for its Undo and Redo. */
  history: { past: readonly CustomizeHistoryEntry[]; future: readonly CustomizeHistoryEntry[] };
  /** The workspace the project is on. */
  presetId: LayoutPresetId;
}

/** A state of the session to step back or forward to. */
export interface CustomizeHistoryEntry {
  draft: CustomizeDraft;
  structure: WorkspaceStructure;
  view: WorkspaceView | undefined;
  resetPending?: true;
  mergeKey?: string;
  mergedAt: string;
}

/** Keyed by project id. */
export type CustomizeSessions = Readonly<Record<string, CustomizeSession>>;

interface ProjectOnWorkspace {
  id: string;
  workspaceId: LayoutPresetId;
}

/** The project's session while it is still on the workspace it customizes, else null. */
export const getCustomizeSession = (
  sessions: CustomizeSessions | undefined,
  project: ProjectOnWorkspace
): CustomizeSession | null => {
  const session = sessions?.[project.id];

  return session?.presetId === project.workspaceId ? session : null;
};

/**
 * Drops the sessions of projects that closed or left their workspace, and of workspaces that no longer exist, so
 * Customize never outlives its workspace. Unchanged sessions keep their identity.
 */
export const pruneCustomizeSessions = (
  sessions: CustomizeSessions | undefined,
  projects: readonly ProjectOnWorkspace[],
  hasWorkspace: (presetId: LayoutPresetId) => boolean
): CustomizeSessions | undefined => {
  if (!sessions) {
    return sessions;
  }

  const kept = Object.entries(sessions).filter(([projectId, session]) => {
    const project = projects.find((candidate) => candidate.id === projectId);

    return project !== undefined && project.workspaceId === session.presetId && hasWorkspace(session.presetId);
  });

  return kept.length === Object.keys(sessions).length ? sessions : Object.fromEntries(kept);
};

/** Whether a workspace's structure (which widgets sit where, and the workspace itself) may change right now. */
export interface StructureLock {
  canEditStructure: boolean;
  /** The active project is customizing its workspace: navigation stays inside it. */
  isCustomizing: boolean;
}

/**
 * The one judge of the lock: structure is editable while the active project customizes its workspace, or everywhere
 * for an account that rearranges without Customize. View changes (fronting, collapsing, resizing, moving windows) are
 * never locked.
 */
export const getStructureLock = (isCustomizing: boolean, rearrangeWithoutCustomize: boolean): StructureLock => ({
  canEditStructure: rearrangeWithoutCustomize || isCustomizing,
  isCustomizing,
});

/** The lock for the snapshot's active project. */
export const selectStructureLock = (
  {
    activeProject,
    customizing,
  }: { activeProject: Parameters<typeof getCustomizeSession>[1]; customizing?: CustomizeSessions },
  rearrangeWithoutCustomize: boolean
): StructureLock =>
  getStructureLock(getCustomizeSession(customizing, activeProject) !== null, rearrangeWithoutCustomize);

export const CUSTOMIZE_WORKSPACE_COMMAND_ID = 'app.customizeWorkspace';
export const RESET_WORKSPACE_VIEW_COMMAND_ID = 'app.resetWorkspaceView';

/** Commands that leave the workspace, or enter Customize again: unavailable while customizing. */
const CUSTOMIZE_EXCLUDED_COMMAND_IDS: ReadonlySet<string> = new Set([
  ...builtInLayoutPresetDescriptors.map(({ hotkeyId }) => `app.${hotkeyId}`),
  'app.backToPreviousWorkspace',
  CUSTOMIZE_WORKSPACE_COMMAND_ID,
  RESET_WORKSPACE_VIEW_COMMAND_ID,
]);

/** Whether the palette offers a command and its hotkey runs it; commands the lock does not concern always are. */
export const isWorkspaceCommandAvailable = (commandId: string, lock: StructureLock): boolean =>
  !(lock.isCustomizing && CUSTOMIZE_EXCLUDED_COMMAND_IDS.has(commandId));
