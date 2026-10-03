import {
  createWorkbenchFocusController,
  type WorkbenchFocusController,
  type WorkbenchFocusScope,
} from './focusRegions';

/** A focus controller for tests that render focus-aware components outside the workbench: one project, every window floating. */
export const createTestFocusController = (scope: Partial<WorkbenchFocusScope> = {}): WorkbenchFocusController =>
  createWorkbenchFocusController({ getProjectId: () => 'project-1', isFloating: () => true, ...scope });
