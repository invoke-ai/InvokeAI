import type { CanvasEditConcurrency } from './editConcurrency';

export const createTestEditConcurrency = (overrides: Partial<CanvasEditConcurrency> = {}): CanvasEditConcurrency => ({
  canEdit: () => true,
  capturePermit: () => ({ epoch: 0 }),
  getEditRevision: () => 0,
  isGestureActive: () => false,
  isPermitCurrent: () => true,
  projectId: 'p',
  ...overrides,
});
