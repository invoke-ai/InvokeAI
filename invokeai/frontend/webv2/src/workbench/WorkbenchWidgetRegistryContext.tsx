import { createContext, use, useMemo, type ReactNode } from 'react';

import type { RegisteredWidget, WorkbenchRegion } from './types';

interface WorkbenchWidgetRegistryContextValue {
  getWidgetsForRegion: (region: WorkbenchRegion) => RegisteredWidget[];
}

const WorkbenchWidgetRegistryContext = createContext<WorkbenchWidgetRegistryContextValue | null>(null);

export const WorkbenchWidgetRegistryProvider = ({
  children,
  getWidgetsForRegion,
}: {
  children: ReactNode;
  getWidgetsForRegion: (region: WorkbenchRegion) => RegisteredWidget[];
}) => {
  const value = useMemo<WorkbenchWidgetRegistryContextValue>(() => ({ getWidgetsForRegion }), [getWidgetsForRegion]);

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
