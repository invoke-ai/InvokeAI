import type * as libraryModule from '@workbench/projects/library';
import type { ProjectLibrarySnapshot, ProjectSummary } from '@workbench/projects/library';
import type { TFunction } from 'i18next';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { getProjectLibrary, refreshProjectLibrary } = vi.hoisted(() => ({
  getProjectLibrary: vi.fn<() => ProjectLibrarySnapshot>(),
  refreshProjectLibrary: vi.fn(() => Promise.resolve()),
}));

// The real compatibility rule stays; only the library's store and fetch are replaced.
vi.mock('@workbench/projects/library', async (importOriginal) => ({
  ...(await importOriginal<typeof libraryModule>()),
  getProjectLibrary,
  refreshProjectLibrary,
}));

import { PaletteSearchUnavailableError } from './entries';
import { createProjectsProvider } from './projectsProvider';

const searchContext = () => ({ signal: new AbortController().signal });

describe('createProjectsProvider', () => {
  const summary = (id: string, name: string, createdAt: string, updatedAt = createdAt): ProjectSummary => ({
    createdAt,
    id,
    minimumCanvasSchemaVersion: 3,
    name,
    revision: 1,
    updatedAt,
  });
  const readyLibrary = (summaries: ProjectSummary[]): ProjectLibrarySnapshot => ({ status: 'ready', summaries });
  const library = readyLibrary([
    summary('p-forest', 'Forest Cabin Concepts', '2026-03-02T10:00:00.000Z', '2026-10-28T12:00:00.000Z'),
    summary('p-city', 'Neon City Keyframes', '2026-09-15T10:00:00.000Z', '2026-10-29T12:00:00.000Z'),
  ]);
  const interpolatingT = ((key: string, options?: { time?: string }) =>
    options?.time === undefined ? key : `${key}: ${options.time}`) as TFunction;
  const createProvider = ({
    openProject = vi.fn(),
    openProjects = [],
    snapshot = library,
  }: {
    openProject?: (projectId: string, name: string) => void;
    openProjects?: { id: string; name: string }[];
    snapshot?: ProjectLibrarySnapshot;
  } = {}) => createProjectsProvider({ library: snapshot, openProject, openProjects, t: interpolatingT });

  beforeEach(() => {
    getProjectLibrary.mockReset();
    refreshProjectLibrary.mockClear();
    vi.useFakeTimers({ now: new Date('2026-10-30T12:00:00.000Z') });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('matches every name term regardless of case and shows when a closed project was last edited', async () => {
    const entries = await createProvider().search({ text: 'cabin FOREST' }, searchContext());

    expect(entries.map(({ group, subtitle, title }) => ({ group, subtitle, title }))).toEqual([
      { group: 'Projects', subtitle: 'projects.editedRelative: 2 days ago', title: 'Forest Cabin Concepts' },
    ]);
    expect(refreshProjectLibrary).not.toHaveBeenCalled();
  });

  it('excludes projects created outside the date range', async () => {
    const entries = await createProvider({ openProjects: [{ id: 'p-unlisted', name: 'Unlisted draft' }] }).search(
      { range: { from: '2026-09-01' }, text: '' },
      searchContext()
    );

    // The open project the library has not listed yet has no known creation date.
    expect(entries.map((entry) => entry.title)).toEqual(['Neon City Keyframes']);
  });

  it('lists an open project once, under its open name, and marks it already open', async () => {
    const entries = await createProvider({
      openProjects: [
        { id: 'p-city', name: 'Neon City Keyframes v2' },
        { id: 'p-new', name: 'New City Study' },
      ],
    }).search({ text: 'city' }, searchContext());

    expect(entries.map(({ id, subtitle, title }) => ({ id, subtitle, title }))).toEqual([
      { id: 'project:p-new', subtitle: 'commandPalette.providers.projectOpen', title: 'New City Study' },
      { id: 'project:p-city', subtitle: 'commandPalette.providers.projectOpen', title: 'Neon City Keyframes v2' },
    ]);
  });

  it('loads a library that is not ready before searching it', async () => {
    getProjectLibrary.mockReturnValue(library);

    const entries = await createProvider({ snapshot: { status: 'idle', summaries: [] } }).search(
      { text: 'neon' },
      searchContext()
    );

    expect(refreshProjectLibrary).toHaveBeenCalledOnce();
    expect(entries.map((entry) => entry.title)).toEqual(['Neon City Keyframes']);
  });

  it('reports the search unavailable when the library cannot load, and retries the load on the next search', async () => {
    const failed: ProjectLibrarySnapshot = { error: 'offline', status: 'error', summaries: [] };
    getProjectLibrary.mockReturnValue(failed);
    const provider = createProvider({ snapshot: failed });

    await expect(provider.search({ text: 'neon' }, searchContext())).rejects.toEqual(
      new PaletteSearchUnavailableError('commandPalette.states.projectsUnavailable')
    );
    await expect(provider.search({ text: 'neon' }, searchContext())).rejects.toBeInstanceOf(
      PaletteSearchUnavailableError
    );
    expect(refreshProjectLibrary).toHaveBeenCalledTimes(2);
  });

  it('leaves out projects this build cannot open', async () => {
    const entries = await createProvider({
      snapshot: readyLibrary([
        ...library.summaries,
        { ...summary('p-future', 'Neon Future', '2026-10-01T10:00:00.000Z'), minimumCanvasSchemaVersion: 999 },
      ]),
    }).search({ text: 'neon' }, searchContext());

    expect(entries.map((entry) => entry.title)).toEqual(['Neon City Keyframes']);
  });

  it('opens the chosen project by id and name', async () => {
    const openProject = vi.fn();
    const [entry] = await createProvider({ openProject }).search({ text: 'neon' }, searchContext());

    entry?.run();

    expect(openProject).toHaveBeenCalledWith('p-city', 'Neon City Keyframes');
  });

  it('re-keys its results when the library summaries or the open projects change, not on a repeated failure', () => {
    const first = createProvider();

    expect(createProvider().contextKey).toBe(first.contextKey);
    expect(createProvider({ snapshot: { ...library, status: 'error' } }).contextKey).toBe(first.contextKey);
    expect(createProvider({ snapshot: readyLibrary([...library.summaries]) }).contextKey).not.toBe(first.contextKey);
    expect(createProvider({ openProjects: [{ id: 'p-city', name: 'Neon City Keyframes' }] }).contextKey).not.toBe(
      first.contextKey
    );
  });
});
