import type { LayoutPresetId } from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

import { getLayoutPreset, isBuiltInLayoutPresetId } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';

import type { LayoutAvailability, ResolvedLayout } from './resolveLayout';
import type { WorkspaceStructure } from './structure';
import type { CustomizeSessions } from './structureLock';
import type { SurfaceAvailability, WorkspaceView, WorkspaceViews } from './view';
import type { SavedWorkspace } from './workspaceDerivations';

import { getWorkspaceStructure } from './accountWorkspaces';
import { getCustomizeBase } from './customizeDraft';
import { resolveLayout } from './resolveLayout';
import { isSameStructure } from './structure';
import { getCustomizeSession } from './structureLock';
import { captureLayout } from './structureReducer';

/** What a project's layout is read from: the account's workspaces and the session's Customize drafts. */
export interface LayoutSource {
  account: AccountState;
  customizing?: CustomizeSessions;
}

/** The parts of a project its layout depends on. */
export interface ProjectOnLayout {
  id: string;
  views: WorkspaceViews;
  workspaceId: LayoutPresetId;
}

/** The structure a project shows: its Customize draft while it customizes, else its workspace's in the account. */
export const getProjectStructure = ({ account, customizing }: LayoutSource, project: ProjectOnLayout) =>
  getCustomizeSession(customizing, project)?.structure ?? getWorkspaceStructure(account, project.workspaceId);

/** The key of the view the project shows: a new workspace's draft while it is customized, else its workspace. */
export const getProjectViewId = (source: LayoutSource, project: ProjectOnLayout): LayoutPresetId =>
  getCustomizeSession(source.customizing, project)?.newWorkspaceId ?? project.workspaceId;

export const getProjectView = (source: LayoutSource, project: ProjectOnLayout): WorkspaceView | undefined =>
  project.views[getProjectViewId(source, project)];

/** The workspace a project is on as its derivations read it (stage, job): the draft's while it customizes. */
export const getProjectWorkspace = (source: LayoutSource, project: ProjectOnLayout): SavedWorkspace => {
  const session = getCustomizeSession(source.customizing, project);

  if (!session) {
    return resolveSavedLayoutPreset(source.account, project.workspaceId);
  }

  const base = getCustomizeBase(source.account, session);
  const defaultRoute = session.draft.defaultRoute === undefined ? base?.defaultRoute : session.draft.defaultRoute;

  return { ...(defaultRoute ? { defaultRoute } : {}), snapshot: session.structure };
};

export type LayoutResolver = (
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  availability: LayoutAvailability
) => ResolvedLayout;

/** What the project shows now. Pass a resolver (`createLayoutResolver`) to keep identities across calls. */
export const resolveProjectLayout = (
  source: LayoutSource,
  project: ProjectOnLayout,
  availability: LayoutAvailability,
  resolver: LayoutResolver = resolveLayout
): ResolvedLayout => resolver(getProjectStructure(source, project), getProjectView(source, project), availability);

type ShippedStructureSnapshot = LayoutSource & { activeLayout: ResolvedLayout; activeProject: ProjectOnLayout };

/**
 * A snapshot selector: whether Done would save the built-in being customized with its shipped structure, as the
 * screen shows it now. It recomputes only when the draft, the view or the layout changes.
 */
export const createShippedStructureSelector = (surfaces: SurfaceAvailability) => {
  let last:
    | { answer: boolean; layout: ResolvedLayout; structure: WorkspaceStructure; view: WorkspaceView | undefined }
    | undefined;

  return (snapshot: ShippedStructureSnapshot): boolean => {
    const session = getCustomizeSession(snapshot.customizing, snapshot.activeProject);

    if (!session || session.newWorkspaceId !== undefined || !isBuiltInLayoutPresetId(session.presetId)) {
      return false;
    }

    const view = getProjectView(snapshot, snapshot.activeProject);

    if (last?.structure !== session.structure || last.view !== view || last.layout !== snapshot.activeLayout) {
      last = {
        answer: isSameStructure(
          captureLayout(session.structure, view, snapshot.activeLayout, surfaces),
          getLayoutPreset(session.presetId).snapshot
        ),
        layout: snapshot.activeLayout,
        structure: session.structure,
        view,
      };
    }

    return last.answer;
  };
};
