import { SessionExpiryGuard } from './workbench/auth/components/SessionExpiryGuard';
import { ThemeController } from './workbench/ThemeController';
import { ModelsRuntime } from './workbench/widgets/models/ModelsRuntime';
import { WorkbenchProvider } from './workbench/WorkbenchContext';
import { WorkbenchRuntime } from './workbench/WorkbenchRuntime';
import { WorkbenchShell } from './workbench/WorkbenchShell';

/**
 * The authenticated application: providers, runtimes, and the shell. Mounted
 * by the index route once the auth guard has resolved, so the backend socket
 * always connects with a valid token (or none, in single-user mode).
 */
export const WorkbenchApp = () => (
  <WorkbenchProvider>
    <ThemeController />
    <SessionExpiryGuard />
    <WorkbenchRuntime />
    <ModelsRuntime />
    <WorkbenchShell />
  </WorkbenchProvider>
);
