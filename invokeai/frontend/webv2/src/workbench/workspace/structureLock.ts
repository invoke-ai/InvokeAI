import type { LayoutPresetId, LayoutPresetRoute, LayoutPresetSnapshot } from '@workbench/layoutContracts';

import { builtInLayoutPresetDescriptors } from '@workbench/layoutPresets';

/** Name, icon and route edits made in the Customize bar: Done applies them, Cancel drops them. Absent is unchanged. */
export interface CustomizeDraft {
  /** `null` is Auto: the job follows the fronted sources. */
  defaultRoute?: LayoutPresetRoute | null;
  iconId?: string;
  label?: string;
}

/** One project customizing the workspace it is on. Session-only: never persisted, synced or undone. */
export interface CustomizeSession {
  draft: CustomizeDraft;
  /** The arrangement on screen when Customize began: Cancel lays it out again, and Done saves only a change from it. */
  entry: LayoutPresetSnapshot;
  presetId: LayoutPresetId;
}

/**
 * Whether Reset to default is pending in the project's Customize session: while its step can still be undone. Done then
 * removes the built-in's overrides; Cancel keeps them.
 */
export const isCustomizeResetPending = (project: {
  undoRedo: { past: readonly { customizeReset?: true }[] };
}): boolean => project.undoRedo.past.some((entry) => entry.customizeReset === true);

/** Keyed by project id. */
export type CustomizeSessions = Readonly<Record<string, CustomizeSession>>;

interface ProjectOnWorkspace {
  id: string;
  layout: { presetId: LayoutPresetId };
}

/** The project's session while it is still on the workspace it customizes, else null. */
export const getCustomizeSession = (
  sessions: CustomizeSessions | undefined,
  project: ProjectOnWorkspace
): CustomizeSession | null => {
  const session = sessions?.[project.id];

  return session?.presetId === project.layout.presetId ? session : null;
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

    return project !== undefined && project.layout.presetId === session.presetId && hasWorkspace(session.presetId);
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

/** Commands that write a workspace's structure: offered only while it may be edited. */
const STRUCTURE_COMMAND_IDS: ReadonlySet<string> = new Set(['app.resetPanelLayout', 'app.saveLayoutPreset']);

/** Commands that leave the workspace, or enter Customize again: unavailable while customizing. */
const CUSTOMIZE_EXCLUDED_COMMAND_IDS: ReadonlySet<string> = new Set([
  ...builtInLayoutPresetDescriptors.map(({ hotkeyId }) => `app.${hotkeyId}`),
  'app.backToPreviousWorkspace',
  CUSTOMIZE_WORKSPACE_COMMAND_ID,
]);

export const isStructureCommand = (commandId: string): boolean => STRUCTURE_COMMAND_IDS.has(commandId);

/** Whether the palette offers a command and its hotkey runs it; commands the lock does not concern always are. */
export const isWorkspaceCommandAvailable = (commandId: string, lock: StructureLock): boolean =>
  STRUCTURE_COMMAND_IDS.has(commandId)
    ? lock.canEditStructure
    : !(lock.isCustomizing && CUSTOMIZE_EXCLUDED_COMMAND_IDS.has(commandId));
