import type { ProjectSummary } from '@workbench/projects/library';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const getWorkbenchPreferences = vi.hoisted(() => vi.fn());
const patchWorkbenchPreferences = vi.hoisted(() => vi.fn(() => Promise.resolve()));

vi.mock('@workbench/settings/store', () => ({ getWorkbenchPreferences, patchWorkbenchPreferences }));

const { dropProjectPin, prunePinnedProjects, toggleProjectPinPreference } =
  await import('@workbench/launchpad/projects/projectPins');

const summary = (id: string): ProjectSummary => ({
  createdAt: '2026-08-01T00:00:00.000Z',
  id,
  minimumCanvasSchemaVersion: 2,
  name: id,
  revision: 1,
  updatedAt: '2026-08-01T00:00:00.000Z',
});

const withPins = (launchpadPinnedProjectIds: string[]) => {
  getWorkbenchPreferences.mockReturnValue({ launchpadPinnedProjectIds });
};

describe('project pin preferences', () => {
  beforeEach(() => {
    patchWorkbenchPreferences.mockClear();
    getWorkbenchPreferences.mockReset();
  });

  it('toggles against the live snapshot rather than a captured value', () => {
    withPins(['a']);
    toggleProjectPinPreference('b');

    expect(patchWorkbenchPreferences).toHaveBeenCalledWith({ launchpadPinnedProjectIds: ['a', 'b'] });
  });

  it('drops a pin when its project is deleted', () => {
    withPins(['a', 'b']);
    dropProjectPin('a');

    expect(patchWorkbenchPreferences).toHaveBeenCalledWith({ launchpadPinnedProjectIds: ['b'] });
  });

  it('does not write when the deleted project was never pinned', () => {
    withPins(['a']);
    dropProjectPin('zzz');

    expect(patchWorkbenchPreferences).not.toHaveBeenCalled();
  });

  it('prunes pins whose project no longer exists', () => {
    withPins(['a', 'gone', 'b']);
    prunePinnedProjects([summary('a'), summary('b')]);

    expect(patchWorkbenchPreferences).toHaveBeenCalledWith({ launchpadPinnedProjectIds: ['a', 'b'] });
  });

  it('stays silent when every pin still resolves, so a refresh is not a write', () => {
    withPins(['a', 'b']);
    prunePinnedProjects([summary('a'), summary('b')]);

    expect(patchWorkbenchPreferences).not.toHaveBeenCalled();
  });

  it('skips the snapshot comparison entirely when nothing is pinned', () => {
    withPins([]);
    prunePinnedProjects([]);

    expect(patchWorkbenchPreferences).not.toHaveBeenCalled();
  });
});
