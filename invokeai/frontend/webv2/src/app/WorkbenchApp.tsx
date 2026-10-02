import type { WorkbenchSearch } from '@workbench/projects/session';

import { LLMTaskProgressRuntime } from '@features/generation/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useSearch } from '@tanstack/react-router';
import { preloadBootWidgets } from '@workbench/bootWidgetPreload';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { WorkbenchCommandPalette } from '@workbench/palette/WorkbenchCommandPalette';
import { WorkbenchShell } from '@workbench/shell';
import { WidgetHosts } from '@workbench/widget-frame/WidgetHosts';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { WorkbenchProvider } from '@workbench/WorkbenchContext';
import { WorkbenchRuntime } from '@workbench/WorkbenchRuntime';
import { WorkbenchSessionController } from '@workbench/WorkbenchSessionController';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { useMemo } from 'react';

import { BootWidgetHintController } from './BootWidgetHintController';
import { GalleryRealtimeRuntime } from './GalleryRealtimeRuntime';
import { GenerateWidgetSyncRuntime } from './GenerateWidgetSyncRuntime';
import { QueueRuntimeAdapter } from './QueueRuntimeAdapter';
import { RecallParametersRuntime } from './RecallParametersRuntime';
import { WorkbenchUiPorts } from './workbenchPorts';

/**
 * Editor-only composition. Persistence handles initial ?project/?new; the session controller handles later search
 * changes.
 */
export const WorkbenchApp = () => {
  const search = useSearch({ strict: false }) as WorkbenchSearch;
  const loadOptions = useMemo(
    () => ({ createNew: search.new, openProjectId: search.project }),
    [search.new, search.project]
  );

  // Preload widget chunks during hydration; the provider withholds the shell until hydration finishes.
  useMountEffect(preloadBootWidgets);

  return (
    <WorkbenchProvider loadOptions={loadOptions}>
      <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
        <BootWidgetHintController />
        <GenerateWidgetSyncRuntime />
        <LLMTaskProgressRuntime />
        <RecallParametersRuntime />
        <WorkbenchUiPorts>
          <WorkbenchHotkeyRuntime />
          <WorkbenchCommandPalette />
          <QueueRuntimeAdapter />
          <GalleryRealtimeRuntime />
          <WorkbenchRuntime />
          <WorkbenchSessionController search={search} />
          <WidgetHosts />
          <WorkbenchShell />
        </WorkbenchUiPorts>
      </WorkbenchWidgetRegistryProvider>
    </WorkbenchProvider>
  );
};
