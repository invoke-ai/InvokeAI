import type {
  OpenWorkbenchWidgetOptions,
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetRegion,
  WidgetRuntimeApi,
  WidgetTypeId,
  WorkbenchRegion,
} from '@workbench/types';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchAction } from '@workbench/workbenchState';

import {
  commandApi,
  commandPaletteApi,
  hotkeyApi,
  menuApi,
  searchApi,
  toolbarApi,
} from '@workbench/extensions/extensionApi';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { useMemo, type Dispatch } from 'react';

const WIDGET_REGIONS = new Set<WidgetRegion>(['bottom', 'center', 'left', 'right']);

const isWidgetRegion = (region: WorkbenchRegion): region is WidgetRegion => WIDGET_REGIONS.has(region as WidgetRegion);

export const createWidgetRuntime = ({
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
  instance: WidgetInstanceContract;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
}): WidgetRuntimeApi => ({
  commands: commandApi,
  hotkeys: hotkeyApi,
  instanceId: instance.id,
  menus: menuApi,
  palette: commandPaletteApi,
  patchState: (values) => dispatch({ instanceId: instance.id, type: 'patchWidgetInstanceValues', values }),
  region,
  search: searchApi,
  setState: (values) => dispatch({ instanceId: instance.id, type: 'setWidgetInstanceValues', values }),
  state: instance.state.values,
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
      openWidgetPlacement({ dispatch, getWidgetsForRegion, options, typeId }),
    revealWidgetInstance: (instanceId) => {
      if (!isWidgetRegion(region)) {
        return { ok: false, reason: 'unsupported' };
      }

      return revealWidgetPlacement({ dispatch, instanceId, project, region });
    },
  },
});

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
  instance: WidgetInstanceContract;
  project: WidgetPlacementProject;
  region: WorkbenchRegion;
}): WidgetRuntimeApi =>
  useMemo(
    () => createWidgetRuntime({ dispatch, getWidgetById, getWidgetsForRegion, instance, project, region }),
    [dispatch, getWidgetById, getWidgetsForRegion, instance, project, region]
  );
