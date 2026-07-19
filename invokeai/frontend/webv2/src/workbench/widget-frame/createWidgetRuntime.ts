import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type {
  WidgetHotkeyApi,
  OpenWorkbenchWidgetOptions,
  RegisteredWidget,
  WidgetContributionSource,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeStateApi,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchWidgetCommands } from '@workbench/workbenchStore';

import { createProjectLogger } from '@workbench/diagnostics/logger';
import {
  commandApi as sharedCommandApi,
  commandPaletteApi,
  hotkeyApi,
  menuApi,
  searchApi,
  toolbarApi,
} from '@workbench/extensions/extensionApi';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

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
  getWidgetsForRegion,
  getWidgetById,
  instance,
  project,
  region,
  state,
  widgets,
}: {
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  instance: WidgetInstanceRuntimeMeta;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
  state: WidgetRuntimeStateApi;
  widgets: WorkbenchWidgetCommands;
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
  const diagnostics = {
    logger: (namespace) => createProjectLogger(namespace, { ...source, kind: 'widget' }),
  } satisfies WidgetRuntimeApi['diagnostics'];

  return {
    commands,
    diagnostics,
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

        return closeWidgetPlacement({ getWidgetById, instanceId, project, region, widgets });
      },
      openWidget: (typeId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions) =>
        openWidgetPlacement({ getWidgetsForRegion, options, projectId: project.projectId, typeId, widgets }),
      revealWidgetInstance: (instanceId) => {
        if (!isWidgetRegion(region)) {
          return { ok: false, reason: 'unsupported' };
        }

        return revealWidgetPlacement({ instanceId, project, region, widgets });
      },
    },
  };
};

export const useWidgetRuntime = ({
  getWidgetsForRegion,
  getWidgetById,
  instance,
  project,
  region,
}: {
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  instance: WidgetInstanceRuntimeMeta;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
}): WidgetRuntimeApi => {
  const store = useWorkbenchInternalStore();
  const { widgets } = store.commands;
  const projectId = project.projectId ?? store.getSnapshot().activeProject.id;
  const state = useMemo<WidgetRuntimeStateApi>(
    () => ({
      getSnapshot: () => getProjectWidgetRuntimeState(store.getState().projects, projectId, instance.id),
      patch: (values) => widgets.patchInstanceValues(instance.id, cloneWidgetRuntimeState(values), projectId),
      set: (values) => widgets.setInstanceValues(instance.id, cloneWidgetRuntimeState(values), projectId),
    }),
    [instance.id, projectId, store, widgets]
  );

  return useMemo(
    () => createWidgetRuntime({ getWidgetById, getWidgetsForRegion, instance, project, region, state, widgets }),
    [getWidgetById, getWidgetsForRegion, instance, project, region, state, widgets]
  );
};
