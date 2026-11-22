import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WorkbenchState } from '@workbench/projectContracts';
import type {
  WidgetHotkeyApi,
  WidgetContributionSource,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeStateApi,
  WidgetRuntimeApi,
  WorkbenchRegion,
} from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchWidgetCommands } from '@workbench/workbenchStore';
import type { OpenInWorkspace } from '@workbench/workspace/workspaceNavigation';

import { createLogger } from '@platform/logging/logger';
import { isWidgetRegion as isKnownWidgetRegion } from '@workbench/layoutContracts';
import { revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { resolveWidgetValues } from '@workbench/widgetState';
import { useWorkbenchExtensions, useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { useMemo } from 'react';

const isWidgetRegion = (region: WorkbenchRegion): region is WidgetRegion => isKnownWidgetRegion(region);

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

/** A copy of the instance's values as it reads them: its pins over the user's defaults for its type. */
export const getProjectWidgetRuntimeState = <State extends Record<string, unknown>>(
  { projects, scopedDefaults }: Pick<WorkbenchState, 'projects' | 'scopedDefaults'>,
  projectId: string,
  instanceId: string
): State =>
  cloneWidgetRuntimeState(
    resolveWidgetValues(
      projects.find((project) => project.id === projectId)?.widgetInstances[instanceId],
      scopedDefaults
    ) as State
  );

export const createWidgetRuntime = ({
  extensions,
  instance,
  open,
  project,
  region,
  state,
  widgets,
}: {
  extensions: ExtensionRegistry;
  instance: WidgetInstanceRuntimeMeta;
  open: OpenInWorkspace;
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
    execute: (commandId, ...args) => extensions.commands.executeForSource(commandId, source, ...args),
    executeForSource: extensions.commands.executeForSource,
    register: (command) => extensions.commands.register({ ...command, source }),
  } satisfies WidgetRuntimeApi['commands'];
  const hotkeys: WidgetHotkeyApi = {
    register: (hotkey) =>
      extensions.hotkeys.register({
        ...hotkey,
        scope: hotkey.scope ?? 'widget',
        source,
      }),
  };
  const diagnostics = {
    logger: (namespace) =>
      createLogger({
        area: instance.typeId,
        namespace,
        projectId: sourceProjectId,
        widget: { instanceId: instance.id, region, typeId: instance.typeId },
      }),
  } satisfies WidgetRuntimeApi['diagnostics'];

  return {
    commands,
    diagnostics,
    hotkeys,
    instanceId: instance.id,
    palette: {
      register: (entry) => extensions.palette.register({ ...entry, source }),
    },
    region,
    search: {
      registerProvider: (provider) => extensions.search.registerProvider({ ...provider, source }),
    },
    state,
    typeId: instance.typeId,
    workbench: {
      open: (intent) => open(intent),
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
  instance,
  project,
  region,
}: {
  instance: WidgetInstanceRuntimeMeta;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
}): WidgetRuntimeApi => {
  const store = useWorkbenchInternalStore();
  const extensions = useWorkbenchExtensions();
  const open = useWorkspaceOpen();
  const { widgets } = store.commands;
  const projectId = project.projectId ?? store.getSnapshot().activeProject.id;
  const state = useMemo<WidgetRuntimeStateApi>(
    () => ({
      getSnapshot: () => getProjectWidgetRuntimeState(store.getState(), projectId, instance.id),
      patch: (values) => widgets.patchInstanceValues(instance.id, cloneWidgetRuntimeState(values), projectId),
      set: (values) => widgets.setInstanceValues(instance.id, cloneWidgetRuntimeState(values), projectId),
    }),
    [instance.id, projectId, store, widgets]
  );

  return useMemo(
    () =>
      createWidgetRuntime({
        extensions,
        instance,
        open,
        project,
        region,
        state,
        widgets,
      }),
    [extensions, instance, open, project, region, state, widgets]
  );
};
