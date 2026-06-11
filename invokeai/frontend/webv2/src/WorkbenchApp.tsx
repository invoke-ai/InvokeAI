import { useSearch } from '@tanstack/react-router';

import { SessionExpiryGuard } from './workbench/auth/components/SessionExpiryGuard';
import { ModelsRuntime } from './workbench/widgets/models/ModelsRuntime';
import { WorkbenchProvider } from './workbench/WorkbenchContext';
import { WorkbenchRuntime } from './workbench/WorkbenchRuntime';
import { WorkbenchSessionController } from './workbench/WorkbenchSessionController';
import { WorkbenchShell } from './workbench/WorkbenchShell';
import type { WorkbenchSearch } from './workbench/projects/session';

/**
 * The authenticated editor: providers, runtimes, and the shell. Mounted by
 * the /app route once the auth guard has resolved, so the backend socket
 * always connects with a valid token (or none, in single-user mode).
 *
 * The route's search params shape the boot: ?project deep-links a library
 * project into the session, ?new starts a fresh draft. Both are consumed by
 * the persistence load; the session controller handles params that change
 * while the editor is already mounted.
 */
export const WorkbenchApp = () => {
  const search = useSearch({ strict: false }) as WorkbenchSearch;

  return (
    <WorkbenchProvider loadOptions={{ createNew: search.new, openProjectId: search.project }}>
      <SessionExpiryGuard />
      <WorkbenchRuntime />
      <ModelsRuntime />
      <WorkbenchSessionController search={search} />
      <WorkbenchShell />
    </WorkbenchProvider>
  );
};
