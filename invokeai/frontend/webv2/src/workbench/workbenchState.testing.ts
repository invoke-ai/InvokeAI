/** Test-only reducer facade; production callers use Workbench commands and queries. */
import type { LayoutPresetId, PanelState, WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';
import type { StoredProjectLayout } from '@workbench/workspace/migrateProjectLayout';
import type { ResolvedLayout } from '@workbench/workspace/resolveLayout';

import { captureInvocationRuntimeInputs } from './invocationPreparation';
import {
  createInitialWorkbenchState,
  __workbenchReducerInternal,
  nextLayerName,
  normalizeWorkbenchProject,
  type __WorkbenchReducerActionInternal,
  type WorkbenchReducerContext,
} from './workbenchState';
import { getWorkspaceStructure } from './workspace/accountWorkspaces';
import { resolveProjectLayout } from './workspace/projectLayout';
import { createRegistryLayoutAvailability } from './workspace/registryAvailability';

type Submission = Extract<__WorkbenchReducerActionInternal, { runtime: unknown }>;
type OptionalRuntime<T> = T extends Submission ? Omit<T, 'runtime'> & { runtime?: T['runtime'] } : T;
type TestAction = OptionalRuntime<__WorkbenchReducerActionInternal>;

const DEFAULT_TEST_CONTEXT: WorkbenchReducerContext = {
  autoSwitchInvocationRoute: true,
  availability: createRegistryLayoutAvailability(),
};

export const getActiveProject = (state: WorkbenchState): Project =>
  state.projects.find(({ id }) => id === state.activeProjectId)!;

/** What a project shows in `state` (the active one by default), as the store resolves it. */
export const getLayoutOf = (state: WorkbenchState, project: Project = getActiveProject(state)): ResolvedLayout =>
  resolveProjectLayout(state, project, DEFAULT_TEST_CONTEXT.availability);

export const workbenchReducer = (
  state: WorkbenchState,
  action: TestAction,
  context: WorkbenchReducerContext = DEFAULT_TEST_CONTEXT
): WorkbenchState =>
  __workbenchReducerInternal(
    state,
    action.type === 'submitResolvedInvocationSnapshot' ||
      action.type === 'submitCanvasInvocationSnapshot' ||
      action.type === 'submitModelOnlyUpscale'
      ? ({
          ...action,
          runtime:
            action.runtime ??
            captureInvocationRuntimeInputs(
              action.type === 'submitResolvedInvocationSnapshot'
                ? action.route.sourceId
                : action.type === 'submitCanvasInvocationSnapshot'
                  ? 'canvas'
                  : 'upscale'
            ),
        } as __WorkbenchReducerActionInternal)
      : (action as __WorkbenchReducerActionInternal),
    context
  );

/** A custom workspace made as the app makes one: New workspace drafts what the active project shows, then Done. */
export const addCustomWorkspace = (
  state: WorkbenchState,
  { iconId, id, label }: { iconId?: string; id: LayoutPresetId; label: string }
): WorkbenchState =>
  workbenchReducer(
    workbenchReducer(state, {
      newWorkspace: { defaultRoute: null, id, label, ...(iconId === undefined ? {} : { iconId }) },
      type: 'beginCustomize',
    }),
    { type: 'finishCustomize' }
  );

export { createInitialWorkbenchState, nextLayerName };
export type { TestAction as WorkbenchAction, WorkbenchReducerContext };

/**
 * `project` as a build before structure and view split stored it, loaded as a project record is: its layout fields
 * validated and carried, waiting for the reducer to migrate them. `storedAt` is the record's last save.
 */
export const asStoredBeforeSplit = (
  project: Project,
  { floatingWidgets, heldView, layout, presetWorkingLayouts, storedAt, widgetRegions }: StoredProjectLayout
): Project => {
  const { legacyLayout: _legacy, views: _views, workspaceId: _workspaceId, ...rest } = project;
  const loaded = normalizeWorkbenchProject({
    ...rest,
    ...(floatingWidgets ? { floatingWidgets } : {}),
    ...(heldView ? { heldView } : {}),
    layout,
    ...(presetWorkingLayouts ? { presetWorkingLayouts } : {}),
    widgetRegions,
  } as unknown as Project);

  return storedAt === undefined ? loaded : { ...loaded, legacyLayout: { ...loaded.legacyLayout!, storedAt } };
};

/**
 * `state` with a built-in workspace arranged as `regions` say (each region's first id in front, every region shown),
 * the way an account that customized it stores it. Instance ids are typed by `types`, else by the part before `:`;
 * each project gets values for the instances it lacks.
 */
export const withArrangement = (
  state: WorkbenchState,
  workspaceId: Extract<LayoutPresetId, 'automate' | 'compose' | 'edit' | 'video'>,
  regions: Partial<Record<WidgetRegion, readonly WidgetInstanceId[]>>,
  types: Readonly<Record<WidgetInstanceId, WidgetTypeId>> = {}
): WorkbenchState => {
  const saved = getWorkspaceStructure(state.account, workspaceId);
  const typeOf = (instanceId: WidgetInstanceId) => types[instanceId] ?? instanceId.split(':', 1)[0]!;
  const widgetRegions = { ...saved.widgetRegions };
  const widgetInstances = { ...saved.widgetInstances };

  for (const [region, instanceIds] of Object.entries(regions) as [WidgetRegion, readonly WidgetInstanceId[]][]) {
    widgetRegions[region] = {
      ...widgetRegions[region],
      activeInstanceId: instanceIds[0] ?? '',
      instanceIds: [...instanceIds],
      isCollapsed: false,
    };
    for (const instanceId of instanceIds) {
      widgetInstances[instanceId] ??= { id: instanceId, typeId: typeOf(instanceId) };
    }
  }

  const panels: PanelState = { isBottomOpen: true, isLeftOpen: true, isRightOpen: true };
  const instanceIds = Object.values(regions).flat();

  return {
    ...state,
    account: {
      ...state.account,
      layoutPresetOverrides: {
        ...state.account.layoutPresetOverrides,
        [workspaceId]: { ...saved, layout: { ...saved.layout, panels }, widgetInstances, widgetRegions },
      },
    },
    projects: state.projects.map((project) => {
      const missing = instanceIds.filter((instanceId) => !project.widgetInstances[instanceId]);

      return missing.length === 0
        ? project
        : {
            ...project,
            widgetInstances: {
              ...project.widgetInstances,
              ...Object.fromEntries(
                missing.map((instanceId) => {
                  const typeId = typeOf(instanceId);
                  const template = Object.values(project.widgetInstances).find(
                    (instance) => instance.typeId === typeId
                  );

                  return [
                    instanceId,
                    template
                      ? { ...template, id: instanceId }
                      : {
                          createdAt: 'test',
                          id: instanceId,
                          state: { id: typeId, label: typeId, values: {}, version: 1 },
                          typeId,
                        },
                  ];
                })
              ),
            },
          };
    }),
  };
};
