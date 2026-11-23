import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { LayoutPreset, LayoutPresetId, WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchNotificationAction } from '@workbench/projectContracts';
import type { RegisteredWidget, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchCommands, WorkbenchQueries, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { getAvailableSourceIdForWidgetTypeId, getWidgetTypeIdForSourceId } from '@workbench/invocationSources';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { doesLayoutPresetExist } from '@workbench/layoutPresetSnapshots';

import type { OpenIntent, OpenResult } from './openIntent';
import type { OpenDecision, OpenTarget, ResolveOpenContext, WorkspaceCandidate } from './resolveOpen';
import type { NewWorkspaceDraft } from './structureLock';
import type { GuestSurface } from './view';
import type { SavedWorkspace, WorkspacePlacement } from './workspaceDerivations';

import { getProjectWorkspace } from './projectLayout';
import { getRegionPresentation, getWidgetNeeds } from './registryAvailability';
import { resolveOpen } from './resolveOpen';
import { getCustomizeSession, isWorkspaceCommandAvailable, selectStructureLock } from './structureLock';
import { getAwayStage, getCenterFront, getPlacedStage, getWorkspaceJobSource } from './workspaceDerivations';

export interface WorkspaceNavigationPorts {
  commands: Pick<WorkbenchCommands, 'layout' | 'notifications' | 'widgets'>;
  /** Moves focus to what a gesture showed; omitted, nothing takes focus. */
  focus?: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  queries: Pick<WorkbenchQueries, 'getSnapshot' | 'getWorkspaceLayout'>;
}

export interface OpenOptions {
  /**
   * Work that needs the widget on screen (setting its route, focusing a field in it): runs in the same task when it
   * shows here, once a cold workspace entry lands, and never when the entry is dropped or nothing showed.
   */
  afterShown?: () => void;
  /**
   * A socket event, finished run or other non-gesture: shows only what this workspace places, moves no focus and
   * shows no guest; its own notice offers what this workspace lacks (`getWorkspaceOffer`).
   */
  background?: boolean;
  /** Which intent's widget takes focus: the first by default, another by its type id, or none. */
  focus?: boolean | WidgetTypeId;
  /** 'stay' never leaves this workspace. */
  mode?: 'navigate' | 'stay';
}

export type OpenInWorkspace = (intents: OpenIntent | readonly OpenIntent[], options?: OpenOptions) => OpenResult;

/**
 * Where each project's last navigation came from and went, with the undo step that switch made, so a switch made by
 * hand since (or undoing that one) is recognizable; session-only.
 */
const lastNavigations = new Map<string, { from: LayoutPresetId; stepId: string | undefined; to: LayoutPresetId }>();

registerAccountOwnedResource({ clear: () => lastNavigations.clear(), name: 'workspace-back' });

/** A preset as the resolver ranks it: what it lays out and the source its job runs. */
export const createWorkspaceCandidate = (
  preset: SavedWorkspace & Pick<LayoutPreset, 'id'>,
  arrangement: WorkspacePlacement
): WorkspaceCandidate => {
  const jobSource = getWorkspaceJobSource(preset, arrangement, getAvailableSourceIdForWidgetTypeId);
  const stage = getPlacedStage(preset, arrangement);

  return {
    arrangement,
    id: preset.id,
    jobSourceTypeId: jobSource ? getWidgetTypeIdForSourceId(jobSource) : null,
    stageTypeId: stage === null ? null : (arrangement.widgetInstances[stage]?.typeId ?? null),
  };
};

const createContext = (ports: WorkspaceNavigationPorts, mode: ResolveOpenContext['mode']): ResolveOpenContext => {
  const { account, activeLayout, activeProject } = ports.queries.getSnapshot();

  return {
    currentWorkspaceId: activeProject.workspaceId,
    getNeeds: getWidgetNeeds(ports.getWidgetsForRegion),
    getPresentation: getRegionPresentation(ports.getWidgetsForRegion),
    here: activeLayout,
    mode,
    workspaces:
      mode === 'stay'
        ? []
        : getOrderedLayoutPresets(account).map((preset) =>
            createWorkspaceCandidate(preset, ports.queries.getWorkspaceLayout(preset.id))
          ),
  };
};

const isShown = (decision: OpenDecision): decision is Extract<OpenDecision, { step: 'hidden' | 'visible' }> =>
  decision.step === 'visible' || decision.step === 'hidden';

/** Whether this workspace can show the intent without leaving: what a background source may front. */
export const canShowHere = (ports: WorkspaceNavigationPorts, intent: OpenIntent): boolean =>
  isShown(resolveOpen(intent, createContext(ports, 'stay')));

/** The action a background source's notice offers when this workspace cannot show its widget: the gesture to it. */
export const getWorkspaceOffer = (
  ports: WorkspaceNavigationPorts,
  intent: OpenIntent
): readonly WorkbenchNotificationAction[] =>
  canShowHere(ports, intent) ? [] : [{ kind: 'open-widget', widgetId: intent.typeId }];

const getTargetRegion = (target: OpenTarget): WidgetRegion | undefined =>
  target.kind === 'region' ? target.region : undefined;

/** Fronts what the decision names, writing view only. */
const show = (ports: WorkspaceNavigationPorts, project: Project, target: OpenTarget, isVisible: boolean): void => {
  if (target.kind === 'floating') {
    ports.commands.widgets.revealFloating(target.instanceId);
  } else if (!isVisible) {
    ports.commands.widgets.front({ instanceId: target.instanceId, projectId: project.id, region: target.region });
  }
};

const focusTarget = (ports: WorkspaceNavigationPorts, typeId: WidgetTypeId, target: OpenTarget): void => {
  if (target.kind === 'floating') {
    ports.focus?.focusFloating(target.instanceId);
  } else {
    ports.focus?.focusRegion(target.region, typeId, target.instanceId);
  }
};

/** The guest of a type the active layout shows, once showing it landed. */
const findGuest = (ports: WorkspaceNavigationPorts, typeId: WidgetTypeId): WidgetInstanceId | undefined => {
  const { activeLayout } = ports.queries.getSnapshot();

  return Object.keys(activeLayout.guests).find((id) => activeLayout.widgetInstances[id]?.typeId === typeId);
};

const getViewport = (): { height: number; width: number } | undefined =>
  typeof window === 'undefined' ? undefined : { height: window.innerHeight, width: window.innerWidth };

/** Shows a widget this workspace does not place as a guest, writing view only; its target, once it shows. */
const showAsGuest = (
  ports: WorkspaceNavigationPorts,
  typeId: WidgetTypeId,
  surface: GuestSurface
): OpenTarget | null => {
  const widget = (['left', 'right', 'bottom', 'center'] as const)
    .flatMap((region) => ports.getWidgetsForRegion(region))
    .find((candidate) => candidate.manifest.id === typeId);

  widget?.implementation.preload();
  ports.commands.widgets.showGuest({
    initialValues: widget?.manifest.state.createInitial(),
    surface,
    typeId,
    viewport: getViewport(),
  });

  const instanceId = findGuest(ports, typeId);

  return instanceId === undefined
    ? null
    : surface === 'center'
      ? { instanceId, kind: 'region', region: 'center' }
      : { instanceId, kind: 'floating' };
};

/** What one intent showed here: where, or null when nothing did. */
type Shown = { decision: OpenDecision; target: OpenTarget | null };

/**
 * Shows a type as this workspace's guest on the asked surface whatever it places, and focuses it: for a gesture that
 * wants one surface the resolver would not pick (the workflow's form while its graph sits in the status bar's panel).
 */
export const openGuest = (ports: WorkspaceNavigationPorts, typeId: WidgetTypeId, surface: GuestSurface): boolean => {
  const target = showAsGuest(ports, typeId, surface);

  if (target) {
    focusTarget(ports, typeId, target);
  }

  return target !== null;
};

/** Shows each intent where this workspace places it, else as a guest (never for a background source), in order. */
const showHere = (
  ports: WorkspaceNavigationPorts,
  intents: readonly OpenIntent[],
  { background, focusIndex }: { background: boolean; focusIndex: number }
): Shown[] =>
  intents.map((intent, index) => {
    const project = ports.queries.getSnapshot().activeProject;
    const decision = resolveOpen(intent, createContext(ports, 'stay'));
    let target: OpenTarget | null = null;

    if (isShown(decision)) {
      show(ports, project, decision.target, decision.step === 'visible');
      target = decision.target;
    } else if (decision.step === 'guest' && !background) {
      target = showAsGuest(ports, intent.typeId, decision.surface);
    }

    if (target && index === focusIndex) {
      focusTarget(ports, intent.typeId, target);
    }

    return { decision, target };
  });

/**
 * The one way to open, show, send to or recall something: shows it where this workspace places it, else enters the
 * workspace that does (first intent only, so one gesture crosses at most one workspace), else shows it here as a
 * guest. Never adds, moves, floats or removes a widget.
 */
export const openInWorkspace = (
  ports: WorkspaceNavigationPorts,
  intents: OpenIntent | readonly OpenIntent[],
  options: OpenOptions = {}
): OpenResult => {
  const list: readonly OpenIntent[] = Array.isArray(intents) ? intents : [intents as OpenIntent];
  const background = options.background === true;
  const focusIndex =
    background || options.focus === false
      ? -1
      : typeof options.focus === 'string'
        ? list.findIndex(({ typeId }) => typeId === options.focus)
        : 0;
  const snapshot = ports.queries.getSnapshot();
  // Customizing keeps navigation inside the workspace until Done or Cancel.
  const mode =
    background || getCustomizeSession(snapshot.customizing, snapshot.activeProject)
      ? 'stay'
      : (options.mode ?? 'navigate');
  const [first] = list;

  if (!first) {
    return { ok: false, reason: 'unavailable' };
  }

  const decision = resolveOpen(first, createContext(ports, mode));

  if (decision.step !== 'elsewhere') {
    const project = ports.queries.getSnapshot().activeProject;
    // An unavailable first intent leaves the rest of the gesture to show what it can.
    const [shown] = showHere(ports, list, { background, focusIndex });

    if (!shown?.target) {
      return { ok: false, reason: 'unavailable' };
    }

    options.afterShown?.();

    return {
      changedWorkspace: false,
      ok: true,
      region: getTargetRegion(shown.target),
      workspaceId: project.workspaceId,
    };
  }

  const { activeProject } = ports.queries.getSnapshot();
  const projectId = activeProject.id;
  const from = activeProject.workspaceId;
  let hasArrived = false;
  const arrive = () => {
    if (hasArrived) {
      return;
    }

    hasArrived = true;
    const entered = ports.queries.getSnapshot().activeProject;

    lastNavigations.set(projectId, { from, stepId: entered.undoRedo.past.at(-1)?.id, to: decision.workspaceId });

    if (showHere(ports, list, { background, focusIndex })[0]?.target) {
      options.afterShown?.();
    }
  };

  // Entering unmounts the editors this workspace shows; their drafts commit first.
  flushWorkbenchDrafts();
  const activation = ports.commands.layout.activatePreset(decision.workspaceId);
  const { activeProject: entered } = ports.queries.getSnapshot();

  if (entered.id === projectId && entered.workspaceId === decision.workspaceId) {
    arrive();
  } else {
    void activation.then((appliedId) => {
      if (appliedId === decision.workspaceId) {
        arrive();
      }
    });
  }

  return {
    changedWorkspace: true,
    ok: true,
    region: getTargetRegion(decision.target),
    workspaceId: decision.workspaceId,
  };
};

/** A switch since the navigation's own step, made by hand or by an undo of it, forgets where it came from. */
const isStillWhereNavigationLanded = (
  project: Project,
  navigation: { stepId: string | undefined; to: LayoutPresetId }
): boolean => {
  if (project.workspaceId !== navigation.to || navigation.stepId === undefined) {
    return false;
  }

  const index = project.undoRedo.past.findIndex(({ id }) => id === navigation.stepId);

  return index >= 0 && project.undoRedo.past.slice(index + 1).every(({ notice }) => notice?.kind !== 'switchPreset');
};

/**
 * Returns to the workspace this project's last navigation left, laid out as it was, while the project is still where
 * that navigation took it. False when there is nothing to go back to.
 */
export const goBackToPreviousWorkspace = (ports: Pick<WorkspaceNavigationPorts, 'commands' | 'queries'>): boolean => {
  const { account, activeProject, customizing } = ports.queries.getSnapshot();
  const navigation = lastNavigations.get(activeProject.id);

  if (getCustomizeSession(customizing, activeProject)) {
    return false;
  }

  lastNavigations.delete(activeProject.id);

  if (
    !navigation ||
    !isStillWhereNavigationLanded(activeProject, navigation) ||
    !doesLayoutPresetExist(account, navigation.from)
  ) {
    return false;
  }

  flushWorkbenchDrafts();
  void ports.commands.layout.activatePreset(navigation.from);

  return true;
};

/** The active workspace's stage while the center shows something else (Away), else null; a snapshot selector. */
export const getActiveAwayStage = (
  snapshot: Pick<WorkbenchSnapshot, 'account' | 'activeLayout' | 'activeProject' | 'customizing'>
): WidgetInstanceId | null =>
  getAwayStage(getProjectWorkspace(snapshot, snapshot.activeProject), snapshot.activeLayout);

/** Ends Away: fronts the stage and keeps focus in the center. False when there is no stage to go back to. */
export const goBackToStage = (ports: WorkspaceNavigationPorts): boolean => {
  const stage = getActiveAwayStage(ports.queries.getSnapshot());

  if (stage === null) {
    return false;
  }

  const { activeLayout, activeProject: project } = ports.queries.getSnapshot();

  ports.commands.widgets.select({ projectId: project.id, region: 'center', widgetId: stage });
  ports.focus?.focusRegion('center', activeLayout.widgetInstances[stage]?.typeId, stage);

  return true;
};

const PREVIEW_TYPE_ID = 'preview';

/**
 * Swaps Preview into the center (Away) and back to the stage, within this workspace. Where Preview is the stage, it
 * swaps to the center's other view and back. A center that does not place Preview shows it as its guest.
 */
export const toggleCenterPreview = (ports: WorkspaceNavigationPorts): boolean => {
  const snapshot = ports.queries.getSnapshot();
  const { activeLayout: layout, activeProject: project } = snapshot;
  const center = layout.widgetRegions.center;
  const previewId = center.instanceIds.find((id) => layout.widgetInstances[id]?.typeId === PREVIEW_TYPE_ID) ?? null;

  if (previewId === null || getCenterFront(layout) !== previewId) {
    return openInWorkspace(ports, { surface: 'stage', typeId: PREVIEW_TYPE_ID }, { focus: false, mode: 'stay' }).ok;
  }

  const stage = getPlacedStage(getProjectWorkspace(snapshot, project), layout);
  const returnId =
    stage !== null && stage !== previewId ? stage : (center.instanceIds.find((id) => id !== previewId) ?? null);

  if (returnId === null) {
    return false;
  }

  ports.commands.widgets.select({ projectId: project.id, region: 'center', widgetId: returnId });

  return true;
};

/**
 * Shows the workflow's form within this workspace: the left rail's, else a window of it, else a guest window. Never the
 * graph, which the resolver answers a panel ask with while it sits in the status bar's panel.
 */
export const showWorkflowForm = (ports: WorkspaceNavigationPorts): void => {
  const { activeLayout: layout, activeProject: project } = ports.queries.getSnapshot();
  const isWorkflow = (instanceId: WidgetInstanceId) => layout.widgetInstances[instanceId]?.typeId === 'workflow';
  const formId = layout.widgetRegions.left.instanceIds.find(isWorkflow);
  // A window shows the surface of the region it returns to: the form for the left rail.
  const formWindow = Object.entries(layout.floatingWidgets ?? {}).find(
    ([instanceId, window]) => isWorkflow(instanceId) && window.returnRegion === 'left'
  )?.[0];

  if (formId) {
    ports.commands.widgets.front({ instanceId: formId, projectId: project.id, region: 'left' });
    ports.focus?.focusRegion('left', undefined, formId);
  } else if (formWindow) {
    ports.commands.widgets.revealFloating(formWindow);
    ports.focus?.focusFloating(formWindow);
  } else {
    openGuest(ports, 'workflow', 'window');
  }
};

/**
 * Per project: the control its session's end returns focus to, and what in its bar takes focus once: the bar, or the
 * name field of a new workspace's draft.
 */
const sessionFocus = new Map<
  string,
  { focus: 'bar' | 'name' | null; returnFocus: (() => HTMLElement | null) | null }
>();

registerAccountOwnedResource({ clear: () => sessionFocus.clear(), name: 'workspace-customize-focus' });

type CustomizePorts = Pick<WorkspaceNavigationPorts, 'queries'> & { commands: Pick<WorkbenchCommands, 'layout'> };

export interface StartCustomizeOptions {
  /** Whether focus moves to the bar; an add that enters Customize focuses the added widget instead. */
  focusBar?: boolean;
  /** The control focus returns to when the session ends, while it is on the page; the workspace's tab otherwise. */
  returnFocus?: () => HTMLElement | null;
}

/**
 * "Keep in workspace": puts a guest in the workspace on its surface, in Customize's draft (entering Customize first
 * while structure is locked) or, rearranging without Customize, in the workspace itself.
 */
export const keepGuestInWorkspace = (
  ports: Pick<WorkspaceNavigationPorts, 'queries'> & { commands: Pick<WorkbenchCommands, 'layout' | 'widgets'> },
  instanceId: WidgetInstanceId,
  {
    allowMultiple,
    canEditStructure,
    returnFocus,
  }: { allowMultiple: boolean; canEditStructure: boolean } & Pick<StartCustomizeOptions, 'returnFocus'>
): void => {
  if (!canEditStructure) {
    startCustomizing(ports, { focusBar: false, returnFocus });
  }

  ports.commands.widgets.keepGuest(instanceId, allowMultiple);
};

/** Enters Customize on the active project's workspace. */
export const startCustomizing = (ports: CustomizePorts, options: StartCustomizeOptions = {}): void => {
  sessionFocus.set(ports.queries.getSnapshot().activeProject.id, {
    focus: options.focusBar === false ? null : 'bar',
    returnFocus: options.returnFocus ?? null,
  });
  ports.commands.layout.beginCustomize();
};

/**
 * Enters Customize on `workspaceId`: entering that workspace first when the project is elsewhere, and only once the
 * entry lands (a dropped one starts nothing). Without one, or for a workspace the account lacks, the active one.
 */
export const customizeWorkspaceById = (ports: CustomizePorts, workspaceId?: LayoutPresetId): void => {
  const { account, activeProject } = ports.queries.getSnapshot();

  if (
    workspaceId === undefined ||
    workspaceId === activeProject.workspaceId ||
    !doesLayoutPresetExist(account, workspaceId)
  ) {
    startCustomizing(ports);

    return;
  }

  flushWorkbenchDrafts();
  void ports.commands.layout.activatePreset(workspaceId).then((appliedId) => {
    if (appliedId === workspaceId && ports.queries.getSnapshot().activeProject.id === activeProject.id) {
      startCustomizing(ports);
    }
  });
};

/**
 * What in the bar takes focus once after a gesture entered Customize or made its draft a new workspace; returning to a
 * project already customizing moves no focus.
 */
export const claimCustomizeBarFocus = (projectId: string): 'bar' | 'name' | null => {
  const entry = sessionFocus.get(projectId);
  const focus = entry?.focus ?? null;

  if (entry) {
    entry.focus = null;
  }

  return focus;
};

export const getWorkspaceTab = (presetId: LayoutPresetId): HTMLElement | null =>
  document.querySelector<HTMLElement>(`[role="tab"][data-layout-preset-id="${CSS.escape(presetId)}"]`);

/** The active workspace's tab, once the tabs are on screen. */
export const getActiveWorkspaceTab = (): HTMLElement | null =>
  document.querySelector<HTMLElement>('[role="tab"][data-layout-preset-id][aria-selected="true"]');

/**
 * However a project's session ends (Done, Cancel, its workspace deleted), focus goes back to the control that opened
 * it while that is on the page, else to the active workspace's tab once the tabs are back.
 */
export const restoreFocusAfterCustomize = (projectId: string): void => {
  const opener = sessionFocus.get(projectId)?.returnFocus;

  sessionFocus.delete(projectId);
  requestAnimationFrame(() => {
    const target = opener?.() ?? null;

    (target?.isConnected ? target : getActiveWorkspaceTab())?.focus();
  });
};

const createCustomPresetId = (): string =>
  `custom-layout-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

type NewWorkspace = Omit<NewWorkspaceDraft, 'id'>;

/**
 * New workspace: Customize on a draft copy of what is on screen, for a workspace that exists once Done creates it.
 * Cancel leaves the account, the tabs and the project's workspace as they were. The name field takes focus.
 */
export const customizeNewWorkspace = (
  ports: CustomizePorts,
  workspace: NewWorkspace,
  options: Pick<StartCustomizeOptions, 'returnFocus'> = {}
): void => {
  sessionFocus.set(ports.queries.getSnapshot().activeProject.id, {
    focus: 'name',
    returnFocus: options.returnFocus ?? null,
  });
  ports.commands.layout.beginCustomize({ ...workspace, id: createCustomPresetId() });
};

/** Save as new: the session goes on as a new workspace's draft, and ends where it would have. */
export const saveCustomizeAsNew = (ports: CustomizePorts, workspace: NewWorkspace): void =>
  ports.commands.layout.saveCustomizeAsNew({ ...workspace, id: createCustomPresetId() });

/** A name no workspace in `taken` has, ignoring case: `base`, else `base` numbered from 2. */
export const getUniqueWorkspaceName = (base: string, taken: readonly string[]): string => {
  const names = new Set(taken.map((name) => name.trim().toLocaleLowerCase()));
  let candidate = base;

  for (let index = 2; names.has(candidate.toLocaleLowerCase()); index += 1) {
    candidate = `${base} ${index}`;
  }

  return candidate;
};

/**
 * Runs a workspace command only while the lock allows it, so its hotkey agrees with the palette that leaves it out:
 * leaving a workspace while customizing does nothing.
 */
export const runWorkspaceCommand = (
  ports: Pick<WorkspaceNavigationPorts, 'queries'>,
  { commandId, rearrangeWithoutCustomize }: { commandId: string; rearrangeWithoutCustomize: boolean },
  handler: () => unknown
): unknown =>
  isWorkspaceCommandAvailable(commandId, selectStructureLock(ports.queries.getSnapshot(), rearrangeWithoutCustomize))
    ? handler()
    : undefined;
