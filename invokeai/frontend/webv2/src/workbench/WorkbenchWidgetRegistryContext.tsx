import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';

import { createContext, use, useMemo, type ReactNode } from 'react';

interface WorkbenchWidgetRegistryContextValue {
  getWidgetById: (widgetId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
}

const WorkbenchWidgetRegistryContext = createContext<WorkbenchWidgetRegistryContextValue | null>(null);

export const WorkbenchWidgetRegistryProvider = ({
  children,
  getWidgetById,
  getWidgetsForRegion,
}: {
  children: ReactNode;
  getWidgetById: (widgetId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
}) => {
  const value = useMemo<WorkbenchWidgetRegistryContextValue>(
    () => ({ getWidgetById, getWidgetsForRegion }),
    [getWidgetById, getWidgetsForRegion]
  );

  return <WorkbenchWidgetRegistryContext value={value}>{children}</WorkbenchWidgetRegistryContext>;
};

export const useWorkbenchWidgetRegistry = (): WorkbenchWidgetRegistryContextValue => {
  const context = use(WorkbenchWidgetRegistryContext);

  if (!context) {
    throw new Error('useWorkbenchWidgetRegistry must be used within a WorkbenchWidgetRegistryProvider.');
  }

  return context;
};

export const useOptionalWorkbenchWidgetRegistry = (): WorkbenchWidgetRegistryContextValue | null =>
  use(WorkbenchWidgetRegistryContext);
