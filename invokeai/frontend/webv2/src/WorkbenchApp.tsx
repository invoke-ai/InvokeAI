import { useSearch } from '@tanstack/react-router';
import { WorkbenchShell } from '@workbench/shell';
import { useMemo } from 'react';

import type { WorkbenchSearch } from './workbench/projects/session';

import { SessionExpiryGuard } from './workbench/auth/components/SessionExpiryGuard';
import { WorkbenchHotkeyRuntime } from './workbench/hotkeys/WorkbenchHotkeyRuntime';
import { WidgetHosts } from './workbench/widget-frame/WidgetHosts';
import { WorkbenchProvider } from './workbench/WorkbenchContext';
import { WorkbenchRuntime } from './workbench/WorkbenchRuntime';
import { WorkbenchSessionController } from './workbench/WorkbenchSessionController';

/**
 * The authenticated editor: providers, editor-only runtimes, and the shell.
 * The shared backend socket is mounted above this route; `WorkbenchRuntime`
 * attaches the generation queue listeners while the editor is open.
 *
 * The route's search params shape the boot: ?project deep-links a library
 * project into the session, ?new starts a fresh draft. Both are consumed by
 * the persistence load; the session controller handles params that change
 * while the editor is already mounted.
 */
export const WorkbenchApp = () => {
  const search = useSearch({ strict: false }) as WorkbenchSearch;
  const loadOptions = useMemo(
    () => ({ createNew: search.new, openProjectId: search.project }),
    [search.new, search.project]
  );

  return (
    <WorkbenchProvider loadOptions={loadOptions}>
      <SessionExpiryGuard />
      <WorkbenchHotkeyRuntime />
      <WorkbenchRuntime />
      <WorkbenchSessionController search={search} />
      <WidgetHosts />
      <WorkbenchShell />
    </WorkbenchProvider>
  );
};
