import type { ProjectLibrarySnapshot, ProjectSummary } from '@workbench/projects/library';

import {
  DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
  MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
} from '@workbench/canvasSchemaVersion';
import { forgetStartupLaunchProject, settleStartupLaunch, stayOnLaunchpad } from '@workbench/launchpad/startupLaunch';
import { afterEach, describe, expect, it } from 'vitest';

import { createStartupRedirects, type InitialPageLoad } from './startupRedirect';

const summary = (id: string, minimumCanvasSchemaVersion = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION): ProjectSummary => ({
  createdAt: '2026-10-01T00:00:00Z',
  id,
  minimumCanvasSchemaVersion,
  name: id,
  revision: 1,
  updatedAt: '2026-10-01T00:00:00Z',
});

const readyLibrary = (...summaries: ProjectSummary[]): ProjectLibrarySnapshot => ({ status: 'ready', summaries });

const startup = (
  openProjectIds: string[] | null,
  library: ProjectLibrarySnapshot = readyLibrary(),
  load: InitialPageLoad = { hash: '', navigationType: 'navigate' },
  { previousEditorRendered = true }: { previousEditorRendered?: boolean } = {}
) => {
  // Each call is a page load; unless a test says otherwise, the previous load's editor rendered.
  if (previousEditorRendered) {
    settleStartupLaunch();
  }
  const calls = { editorPreloads: 0, libraryLoads: 0, sessionPeeks: 0 };
  const redirects = createStartupRedirects(load, {
    reportSkippedLaunch: () => {
      skippedReports += 1;
    },
    loadProjectLibrary: () => {
      calls.libraryLoads += 1;
      return Promise.resolve(library);
    },
    peekOpenProjectIds: () => {
      calls.sessionPeeks += 1;
      return Promise.resolve(openProjectIds);
    },
    preloadEditor: () => {
      calls.editorPreloads += 1;
    },
  });

  return { calls, redirects };
};

let skippedReports = 0;

afterEach(() => {
  // A test that left a launch opening would make the next one stay home; the editor rendering is what clears it.
  settleStartupLaunch();
  window.localStorage.clear();
  skippedReports = 0;
});

const HOME = { pathname: '/', preload: false };
const EDITOR = { pathname: '/app', preload: false };

describe('startup redirects', () => {
  it('restores the saved session when a launch finds open projects, and hands the proof to the editor entry once', async () => {
    for (const hash of ['', '#']) {
      const { calls, redirects } = startup(['project-2', 'project-1'], readyLibrary(), { hash });

      await expect(redirects.take(HOME, 'recentProject')).resolves.toEqual({ replace: true, search: {}, to: '/app' });
      expect(calls).toEqual({ editorPreloads: 1, libraryLoads: 0, sessionPeeks: 1 });

      await expect(redirects.take(EDITOR, 'recentProject')).resolves.toBeNull();
      expect(redirects.takeOpenSessionProof()).toBe(true);
      expect(redirects.takeOpenSessionProof()).toBe(false);
    }
  });

  it('drops the session proof when another route loads before the editor', async () => {
    const { redirects } = startup(['project-1']);

    await redirects.take(HOME, 'recentProject');
    await redirects.take({ pathname: '/models', preload: false }, 'recentProject');

    expect(redirects.takeOpenSessionProof()).toBe(false);
  });

  it('opens the most recently edited compatible project when the session is empty or not saved yet', async () => {
    for (const openProjectIds of [[], null]) {
      const library = readyLibrary(
        summary('newer-incompatible', MAX_SUPPORTED_CANVAS_SCHEMA_VERSION + 1),
        summary('project-7'),
        summary('project-3')
      );
      const { redirects } = startup(openProjectIds, library);

      await expect(redirects.take(HOME, 'recentProject')).resolves.toEqual({
        replace: true,
        search: { project: 'project-7' },
        to: '/app',
      });
      expect(redirects.takeOpenSessionProof()).toBe(false);
    }
  });

  it('stays on the Launchpad without a compatible project or a loaded library', async () => {
    const libraries: ProjectLibrarySnapshot[] = [
      readyLibrary(),
      readyLibrary(summary('incompatible', MAX_SUPPORTED_CANVAS_SCHEMA_VERSION + 1)),
      { error: 'offline', status: 'error', summaries: [summary('stale')] },
      { status: 'idle', summaries: [] },
    ];

    for (const library of libraries) {
      for (const openProjectIds of [[], null]) {
        const { redirects } = startup(openProjectIds, library);

        await expect(redirects.take(HOME, 'recentProject')).resolves.toBeNull();
      }
    }
  });

  it('does no project work for accounts that open the Launchpad', async () => {
    const { calls, redirects } = startup(['project-1'], readyLibrary(summary('project-1')));

    await expect(redirects.take(HOME, 'launchpad')).resolves.toBeNull();
    expect(calls).toEqual({ editorPreloads: 0, libraryLoads: 0, sessionPeeks: 0 });
  });

  it('treats only opening the bare URL as a launch, not reloading or returning to it', async () => {
    for (const navigationType of ['reload', 'back_forward'] as const) {
      const { redirects } = startup(['project-1'], readyLibrary(), { hash: '', navigationType });

      await expect(redirects.take(HOME, 'recentProject')).resolves.toBeNull();
    }
  });

  it('never overrides a deep link, including an explicit Home', async () => {
    for (const hash of ['#/', '#/models', '#/app?project=project-3']) {
      const { calls, redirects } = startup(['project-1'], readyLibrary(), { hash });

      await expect(redirects.take(HOME, 'recentProject')).resolves.toBeNull();
      expect(calls.sessionPeeks).toBe(0);
    }
  });

  it('keeps the launch through preloads and applies it to the first authenticated arrival only', async () => {
    const { calls, redirects } = startup(['project-1']);

    await expect(redirects.take({ pathname: '/', preload: true }, 'recentProject')).resolves.toBeNull();
    expect(calls.sessionPeeks).toBe(0);

    await expect(redirects.take(HOME, 'recentProject')).resolves.toEqual({ replace: true, search: {}, to: '/app' });
    await expect(redirects.take(HOME, 'recentProject')).resolves.toBeNull();

    // Choosing the most recent project on the Launchpad does not turn the next visit to Home into a launch.
    const launchpadFirst = startup(['project-1']).redirects;
    await expect(launchpadFirst.take(HOME, 'launchpad')).resolves.toBeNull();
    await expect(launchpadFirst.take(HOME, 'recentProject')).resolves.toBeNull();
  });

  it('stays on the Launchpad, once and with a notice, when the previous launch never rendered the editor', async () => {
    // The previous page load redirected into the editor and was gone before the editor rendered.
    await startup(['project-1']).redirects.take(HOME, 'recentProject');

    const next = startup(['project-1'], readyLibrary(), undefined, { previousEditorRendered: false });
    await expect(next.redirects.take(HOME, 'recentProject')).resolves.toBeNull();
    expect(next.calls.sessionPeeks).toBe(0);
    expect(skippedReports).toBe(1);

    // Only that one launch: the one after it tries again.
    await expect(
      startup(['project-1'], readyLibrary(), undefined, { previousEditorRendered: false }).redirects.take(
        HOME,
        'recentProject'
      )
    ).resolves.toEqual({ replace: true, search: {}, to: '/app' });
    expect(skippedReports).toBe(1);
  });

  it('gives up its redirect when the viewer chooses the Launchpad from the splash while it decides', async () => {
    let resolvePeek: (ids: string[]) => void = () => undefined;
    const redirects = createStartupRedirects(
      { hash: '', navigationType: 'navigate' },
      {
        loadProjectLibrary: () => Promise.resolve(readyLibrary(summary('project-1'))),
        peekOpenProjectIds: () =>
          new Promise((resolve) => {
            resolvePeek = resolve;
          }),
        preloadEditor: () => undefined,
        reportSkippedLaunch: () => undefined,
      }
    );

    const decision = redirects.take(HOME, 'recentProject');
    stayOnLaunchpad();
    resolvePeek(['project-1']);

    await expect(decision).resolves.toBeNull();
    // Staying is not a failed open: the next launch reopens the work as usual.
    await expect(
      startup(['project-1'], readyLibrary(), undefined, { previousEditorRendered: false }).redirects.take(
        HOME,
        'recentProject'
      )
    ).resolves.not.toBeNull();
  });

  it('keeps the mark when another route loads before the editor renders, and drops it when that project is deleted', async () => {
    const first = startup([], readyLibrary(summary('project-7')));
    await expect(first.redirects.take(HOME, 'recentProject')).resolves.toEqual({
      replace: true,
      search: { project: 'project-7' },
      to: '/app',
    });
    // The editor failed to load and the viewer went to the Launchpad: the next launch stays home.
    await first.redirects.take({ pathname: '/projects', preload: false }, 'recentProject');
    const skipped = startup([], readyLibrary(summary('project-7')), undefined, { previousEditorRendered: false });
    await expect(skipped.redirects.take(HOME, 'recentProject')).resolves.toBeNull();
    expect(skippedReports).toBe(1);

    // A launch whose project was deleted since can no longer fail on it.
    await startup([], readyLibrary(summary('project-7')), undefined, { previousEditorRendered: false }).redirects.take(
      HOME,
      'recentProject'
    );
    forgetStartupLaunchProject('project-7');
    const afterDelete = startup([], readyLibrary(summary('project-3')), undefined, { previousEditorRendered: false });
    await expect(afterDelete.redirects.take(HOME, 'recentProject')).resolves.not.toBeNull();
    expect(skippedReports).toBe(1);
  });

  it('treats an editor that rendered after a reload as settling the launch the reloaded page left marked', async () => {
    await startup(['project-1']).redirects.take(HOME, 'recentProject');
    // A reload during "Opening project": this page never decided a launch, but its editor rendered.
    settleStartupLaunch();

    const next = startup(['project-1'], readyLibrary(), undefined, { previousEditorRendered: false });
    await expect(next.redirects.take(HOME, 'recentProject')).resolves.not.toBeNull();
    expect(skippedReports).toBe(0);
  });
});
