import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { LayoutPreset, LayoutPresetId, LayoutPresetRoute, WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchNotificationAction } from '@workbench/projectContracts';
import type { RegisteredWidget, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchCommands, WorkbenchQueries, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { getAvailableSourceIdForWidgetTypeId, getWidgetTypeIdForSourceId } from '@workbench/invocationSources';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import {
  doesLayoutPresetExist,
  getLayoutPresetArrangement,
  resolveSavedLayoutPreset,
} from '@workbench/layoutPresetSnapshots';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { canRenderWidgetInRegion } from '@workbench/widgetPlacementCommands';
import i18next from 'i18next';

import type { OpenIntent, OpenResult } from './openIntent';
import type {
  OpenDecision,
  OpenTarget,
  RegionPresentation,
  ResolveOpenContext,
  WorkspaceCandidate,
} from './resolveOpen';
import type { SavedWorkspace, WorkspacePlacement } from './workspaceDerivations';

import { resolveOpen } from './resolveOpen';
import {
  getCustomizeSession,
  isStructureCommand,
  isWorkspaceCommandAvailable,
  selectStructureLock,
} from './structureLock';
import { getAwayStage, getCenterFront, getPlacedStage, getWorkspaceJobSource } from './workspaceDerivations';

export interface WorkspaceNavigationPorts {
  commands: Pick<WorkbenchCommands, 'layout' | 'notifications' | 'widgets'>;
  /** Moves focus to what a gesture showed; omitted, nothing takes focus. */
  focus?: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
}

export interface OpenOptions {
  /**
   * Work that needs the widget on screen (setting its route, focusing a field in it): runs in the same task when it
   * shows here, once a cold workspace entry lands, and never when the entry is dropped or nothing showed.
   */
  afterShown?: () => void;
  /**
   * A socket event, finished run or other non-gesture: shows only what this workspace places and moves no focus. It
   * records no notice; its own notice offers what this workspace lacks (`getWorkspaceOffer`).
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

/** The resolver's region rules for this registry: enabled types the account may use, as a view or a chip. */
export const getRegionPresentation =
  (getWidgetsForRegion: WorkspaceNavigationPorts['getWidgetsForRegion']) =>
  (typeId: WidgetTypeId, region: WidgetRegion): RegionPresentation => {
    const widget = getWidgetsForRegion(region).find((candidate) => candidate.manifest.id === typeId);

    if (widget?.status !== 'enabled') {
      return null;
    }

    return canRenderWidgetInRegion(widget, region) &&
      !(region === 'bottom' && widget.manifest.bottomPanel === 'popover')
      ? 'view'
      : 'chip';
  };

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
  const { account, activeProject } = ports.queries.getSnapshot();

  return {
    currentWorkspaceId: activeProject.layout.presetId,
    getPresentation: getRegionPresentation(ports.getWidgetsForRegion),
    here: activeProject,
    mode,
    workspaces:
      mode === 'stay'
        ? []
        : getOrderedLayoutPresets(account).map((preset) =>
            createWorkspaceCandidate(preset, getLayoutPresetArrangement(activeProject, preset))
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

const NOT_HERE_TITLE_KEYS = {
  nowhere: 'shell.workspace.notHere',
  panel: 'shell.workspace.notInPanel',
  stage: 'shell.workspace.notInCenter',
} as const;

/** Offers the structural add on the surface the open asked for; a repeat coalesces into the newest offer. */
const offerAdd = (
  ports: WorkspaceNavigationPorts,
  intent: OpenIntent,
  decision: Extract<OpenDecision, { step: 'not-placed' }>
): void => {
  const widget = WIDGET_REGIONS.map((region) =>
    ports.getWidgetsForRegion(region).find((candidate) => candidate.manifest.id === intent.typeId)
  ).find((candidate) => candidate !== undefined);

  if (!widget) {
    return;
  }

  const titleKey = NOT_HERE_TITLE_KEYS[decision.placedHere && intent.surface ? intent.surface : 'nowhere'];
  const titleValues = { widget: resolveWidgetLabel(widget.manifest, i18next.t) };

  ports.commands.notifications.add({
    actions: [{ kind: 'add-widget', widgetId: intent.typeId, ...(intent.surface ? { surface: intent.surface } : {}) }],
    category: 'workspace-offer',
    kind: 'info',
    title: i18next.t(titleKey, { ...titleValues, lng: 'en' }),
    titleKey,
    titleValues,
  });
};

/** Shows each intent where this workspace places it, in order, and returns the first one's decision. */
const showHere = (
  ports: WorkspaceNavigationPorts,
  intents: readonly OpenIntent[],
  { background, focusIndex }: { background: boolean; focusIndex: number }
): OpenDecision[] =>
  intents.map((intent, index) => {
    const project = ports.queries.getSnapshot().activeProject;
    const decision = resolveOpen(intent, createContext(ports, 'stay'));

    if (isShown(decision)) {
      show(ports, project, decision.target, decision.step === 'visible');

      if (index === focusIndex) {
        focusTarget(ports, intent.typeId, decision.target);
      }
    } else if (decision.step === 'not-placed' && !background) {
      offerAdd(ports, intent, decision);
    }

    return decision;
  });

/**
 * The one way to open, show, send to or recall something: shows it where this workspace places it, else enters the
 * workspace that does (first intent only, so one gesture crosses at most one workspace), else offers to add it here.
 * Never adds, moves, floats or removes a widget.
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

    if (!shown || !isShown(shown)) {
      return { ok: false, reason: shown?.step === 'not-placed' ? 'not-placed' : 'unavailable' };
    }

    options.afterShown?.();

    return {
      changedWorkspace: false,
      ok: true,
      region: getTargetRegion(shown.target),
      workspaceId: project.layout.presetId,
    };
  }

  const { activeProject } = ports.queries.getSnapshot();
  const projectId = activeProject.id;
  const from = activeProject.layout.presetId;
  let hasArrived = false;
  const arrive = () => {
    if (hasArrived) {
      return;
    }

    hasArrived = true;
    const entered = ports.queries.getSnapshot().activeProject;

    lastNavigations.set(projectId, { from, stepId: entered.undoRedo.past.at(-1)?.id, to: decision.workspaceId });

    if (isShown(showHere(ports, list, { background, focusIndex })[0] ?? { step: 'unavailable' })) {
      options.afterShown?.();
    }
  };

  // Entering unmounts the editors this workspace shows; their drafts commit first.
  flushWorkbenchDrafts();
  const activation = ports.commands.layout.activatePreset(decision.workspaceId);
  const { activeProject: entered } = ports.queries.getSnapshot();

  if (entered.id === projectId && entered.layout.presetId === decision.workspaceId) {
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

/** A switch or revert since the navigation's own step, made by hand or by an undo of it, forgets where it came from. */
const isStillWhereNavigationLanded = (
  project: Project,
  navigation: { stepId: string | undefined; to: LayoutPresetId }
): boolean => {
  if (project.layout.presetId !== navigation.to || navigation.stepId === undefined) {
    return false;
  }

  const index = project.undoRedo.past.findIndex(({ id }) => id === navigation.stepId);

  return (
    index >= 0 &&
    project.undoRedo.past
      .slice(index + 1)
      .every(({ notice }) => notice?.kind !== 'switchPreset' && notice?.kind !== 'revert')
  );
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
export const getActiveAwayStage = ({
  account,
  activeProject,
}: Pick<WorkbenchSnapshot, 'account' | 'activeProject'>): WidgetInstanceId | null =>
  getAwayStage(resolveSavedLayoutPreset(account, activeProject.layout.presetId), activeProject);

/** Ends Away: fronts the stage and keeps focus in the center. False when there is no stage to go back to. */
export const goBackToStage = (ports: WorkspaceNavigationPorts): boolean => {
  const stage = getActiveAwayStage(ports.queries.getSnapshot());

  if (stage === null) {
    return false;
  }

  const project = ports.queries.getSnapshot().activeProject;

  ports.commands.widgets.select({ projectId: project.id, region: 'center', widgetId: stage });
  ports.focus?.focusRegion('center', project.widgetInstances[stage]?.typeId, stage);

  return true;
};

const PREVIEW_TYPE_ID = 'preview';

/**
 * Swaps Preview into the center (Away) and back to the stage, within this workspace. Where Preview is the stage, it
 * swaps to the center's other view and back. A center that does not place Preview gets the offer to add it.
 */
export const toggleCenterPreview = (ports: WorkspaceNavigationPorts): boolean => {
  const { account, activeProject: project } = ports.queries.getSnapshot();
  const center = project.widgetRegions.center;
  const previewId = center.instanceIds.find((id) => project.widgetInstances[id]?.typeId === PREVIEW_TYPE_ID) ?? null;

  if (previewId === null || getCenterFront(project) !== previewId) {
    return openInWorkspace(ports, { surface: 'stage', typeId: PREVIEW_TYPE_ID }, { focus: false, mode: 'stay' }).ok;
  }

  const stage = getPlacedStage(resolveSavedLayoutPreset(account, project.layout.presetId), project);
  const returnId =
    stage !== null && stage !== previewId ? stage : (center.instanceIds.find((id) => id !== previewId) ?? null);

  if (returnId === null) {
    return false;
  }

  ports.commands.widgets.select({ projectId: project.id, region: 'center', widgetId: returnId });

  return true;
};

/** Per project: the control its session's end returns focus to, and whether its bar takes focus when it mounts. */
const sessionFocus = new Map<string, { focusBar: boolean; returnFocus: (() => HTMLElement | null) | null }>();

registerAccountOwnedResource({ clear: () => sessionFocus.clear(), name: 'workspace-customize-focus' });

type CustomizePorts = Pick<WorkspaceNavigationPorts, 'queries'> & { commands: Pick<WorkbenchCommands, 'layout'> };

export interface StartCustomizeOptions {
  /** Save as new: the session goes on in the copy and keeps the control that opened it. */
  continues?: boolean;
  /** Whether focus moves to the bar; an add that enters Customize focuses the added widget instead. */
  focusBar?: boolean;
  /** The control focus returns to when the session ends, while it is on the page; the workspace's tab otherwise. */
  returnFocus?: () => HTMLElement | null;
}

/** Enters Customize on the active project's workspace. */
export const startCustomizing = (ports: CustomizePorts, options: StartCustomizeOptions = {}): void => {
  const projectId = ports.queries.getSnapshot().activeProject.id;
  const previous = sessionFocus.get(projectId);

  sessionFocus.set(projectId, {
    focusBar: options.focusBar !== false,
    returnFocus: options.continues ? (previous?.returnFocus ?? null) : (options.returnFocus ?? null),
  });
  ports.commands.layout.beginCustomize();
};

/** True once after a gesture entered Customize; returning to a project already customizing moves no focus. */
export const claimCustomizeBarFocus = (projectId: string): boolean => {
  const focus = sessionFocus.get(projectId);

  if (!focus?.focusBar) {
    return false;
  }

  focus.focusBar = false;

  return true;
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

/**
 * New workspace: copies the arrangement on screen into a new workspace, moves the project onto it and customizes it
 * there. Also what Save as new does from inside Customize.
 */
export const customizeNewWorkspace = (
  ports: CustomizePorts,
  workspace: { defaultRoute: LayoutPresetRoute | null; iconId?: string; label: string },
  options?: StartCustomizeOptions
): void => {
  ports.commands.layout.createPreset(createCustomPresetId(), workspace.label, workspace.iconId, workspace.defaultRoute);
  startCustomizing(ports, options);
};

const LOCKED_TITLE_KEY = 'shell.workspace.locked';

/**
 * Runs a workspace command only while the lock allows it, so its hotkey agrees with the palette that leaves it out. A
 * locked structural command answers with one Customize hint, a repeat coalescing into it; leaving a workspace while
 * customizing does nothing.
 */
export const runWorkspaceCommand = (
  ports: Pick<WorkspaceNavigationPorts, 'queries'> & { commands: Pick<WorkbenchCommands, 'notifications'> },
  { commandId, rearrangeWithoutCustomize }: { commandId: string; rearrangeWithoutCustomize: boolean },
  handler: () => unknown
): unknown => {
  if (
    isWorkspaceCommandAvailable(commandId, selectStructureLock(ports.queries.getSnapshot(), rearrangeWithoutCustomize))
  ) {
    return handler();
  }
  if (isStructureCommand(commandId)) {
    ports.commands.notifications.add({
      actions: [{ kind: 'customize-workspace' }],
      category: 'workspace-offer',
      kind: 'info',
      title: i18next.t(LOCKED_TITLE_KEY, { lng: 'en' }),
      titleKey: LOCKED_TITLE_KEY,
    });
  }

  return undefined;
};
