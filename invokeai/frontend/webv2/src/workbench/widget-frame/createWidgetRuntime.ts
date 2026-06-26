/* eslint-disable react/react-compiler */
import type {
  Project,
  WidgetHotkeyApi,
  OpenWorkbenchWidgetOptions,
  RegisteredWidget,
  WidgetContributionSource,
  WidgetInstanceRuntimeMeta,
  WidgetRegion,
  WidgetRuntimeStateApi,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/types';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchAction } from '@workbench/workbenchState';

import {
  commandApi as sharedCommandApi,
  commandPaletteApi,
  hotkeyApi,
  menuApi,
  searchApi,
  toolbarApi,
} from '@workbench/extensions/extensionApi';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { useProjectWidgetInstanceValuesSelector, useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useMemo, type Dispatch } from 'react';

const WIDGET_REGIONS = new Set<WidgetRegion>(['bottom', 'center', 'left', 'right']);

const isWidgetRegion = (region: WorkbenchRegion): region is WidgetRegion => WIDGET_REGIONS.has(region as WidgetRegion);

const cloneWidgetRuntimeValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(cloneWidgetRuntimeValue);
  }

  if (value instanceof Map) {
    return new Map(Array.from(value, ([key, item]) => [key, cloneWidgetRuntimeValue(item)]));
  }

  if (value instanceof Set) {
    return new Set(Array.from(value, cloneWidgetRuntimeValue));
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, cloneWidgetRuntimeValue(item)])
    );
  }

  return value;
};

export const cloneWidgetRuntimeState = <State extends Record<string, unknown>>(state: State): State => {
  try {
    return structuredClone(state) as State;
  } catch {
    return cloneWidgetRuntimeValue(state) as State;
  }
};

export const getProjectWidgetRuntimeState = <State extends Record<string, unknown>>(
  projects: readonly Project[],
  projectId: string,
  instanceId: string
): State =>
  cloneWidgetRuntimeState(
    (projects.find((project) => project.id === projectId)?.widgetInstances[instanceId]?.state.values ?? {}) as State
  );

export const createWidgetRuntime = ({
  dispatch,
  getWidgetsForRegion,
  getWidgetById,
  instance,
  project,
  region,
  state,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  instance: WidgetInstanceRuntimeMeta;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
  state: WidgetRuntimeStateApi;
}): WidgetRuntimeApi => {
  const sourceProjectId = project.projectId ?? '';
  const source: WidgetContributionSource = {
    instanceId: instance.id,
    projectId: sourceProjectId,
    region,
    typeId: instance.typeId,
  };
  const commands = {
    execute: (commandId, ...args) => sharedCommandApi.executeForSource(commandId, source, ...args),
    executeForSource: sharedCommandApi.executeForSource,
    register: (command) => sharedCommandApi.register({ ...command, source }),
  } satisfies WidgetRuntimeApi['commands'];
  const hotkeys: WidgetHotkeyApi = {
    register: (hotkey) =>
      hotkeyApi.register({
        ...hotkey,
        scope: hotkey.scope ?? 'widget',
        source,
      }),
  };
  const menus = {
    register: (menu) => menuApi.register({ ...menu, source }),
  } satisfies WidgetRuntimeApi['menus'];

  return {
    commands,
    hotkeys,
    instanceId: instance.id,
    menus,
    palette: commandPaletteApi,
    region,
    search: searchApi,
    state,
    toolbars: toolbarApi,
    typeId: instance.typeId,
    workbench: {
      closeWidgetInstance: (instanceId) => {
        if (!isWidgetRegion(region)) {
          return { ok: false, reason: 'unsupported' };
        }

        return closeWidgetPlacement({ dispatch, getWidgetById, instanceId, project, region });
      },
      openWidget: (typeId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions) =>
        openWidgetPlacement({ dispatch, getWidgetsForRegion, options, projectId: project.projectId, typeId }),
      revealWidgetInstance: (instanceId) => {
        if (!isWidgetRegion(region)) {
          return { ok: false, reason: 'unsupported' };
        }

        return revealWidgetPlacement({ dispatch, instanceId, project, region });
      },
    },
  };
};

export const useWidgetRuntime = ({
  dispatch,
  getWidgetsForRegion,
  getWidgetById,
  instance,
  project,
  region,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  instance: WidgetInstanceRuntimeMeta;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
}): WidgetRuntimeApi => {
  const store = useWorkbenchStore();
  const projectId = project.projectId ?? store.getSnapshot().activeProject.id;
  const state = useMemo<WidgetRuntimeStateApi>(
    () => ({
      getSnapshot: () => getProjectWidgetRuntimeState(store.getState().projects, projectId, instance.id),
      patch: (values) =>
        dispatch({
          instanceId: instance.id,
          projectId,
          type: 'patchWidgetInstanceValues',
          values: cloneWidgetRuntimeState(values),
        }),
      set: (values) =>
        dispatch({
          instanceId: instance.id,
          projectId,
          type: 'setWidgetInstanceValues',
          values: cloneWidgetRuntimeState(values),
        }),
      useSelector<Selected>(
        selector: (state: Record<string, unknown>) => Selected,
        isEqual?: (left: Selected, right: Selected) => boolean
      ): Selected {
        return useProjectWidgetInstanceValuesSelector(projectId, instance.id, selector, isEqual);
      },
    }),
    [dispatch, instance.id, projectId, store]
  );

  return useMemo(
    () => createWidgetRuntime({ dispatch, getWidgetById, getWidgetsForRegion, instance, project, region, state }),
    [dispatch, getWidgetById, getWidgetsForRegion, instance, project, region, state]
  );
};
