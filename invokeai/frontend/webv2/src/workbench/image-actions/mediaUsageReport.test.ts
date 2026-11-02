import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsage, GalleryMediaUsageDocument } from '@features/gallery/mediaUsage';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';

import {
  MEDIA_USAGE_MAX_NAMES_PER_KIND,
  MEDIA_USAGE_MAX_PROJECT_READ_BYTES,
  MEDIA_USAGE_MAX_PROJECT_READS,
  resolveMediaUsage,
  type MediaUsageHost,
} from './mediaUsageReport';

// The server and project reads are the external systems; the narrowing scan (`collectMediaUsage`) runs for real.
const mocks = vi.hoisted(() => ({
  check: vi.fn<(request: unknown, signal: AbortSignal) => Promise<unknown>>(),
  getProject: vi.fn<(projectId: string, maxBytes: number, signal: AbortSignal) => Promise<unknown>>(),
}));

vi.mock('@features/gallery/mediaUsage', () => ({
  checkGalleryMediaUsage: (request: unknown, signal: AbortSignal) => mocks.check(request, signal),
}));
vi.mock('@workbench/projects/api', () => ({
  getProjectWithinSize: (projectId: string, maxBytes: number, signal: AbortSignal) =>
    mocks.getProject(projectId, maxBytes, signal),
}));
vi.mock('@workbench/projects/projectHydration', () => ({
  deserializeProjectRecord: (record: { project: unknown }) => ({ project: record.project, status: 'loaded' }),
}));

const image = (name: string): GalleryItemRef => ({ kind: 'image', name });

/** A project whose canvas raster layers hold `raster`; `selected` is only the Gallery's selection. */
const project = (id: string, name: string, { raster = [] as string[], selected = null as string | null } = {}) => ({
  canvas: {
    document: {
      stacks: {
        control: [],
        inpaint_mask: [],
        raster: raster.map((imageName, index) => ({ id: `${id}-${index}`, source: { image: { imageName } } })),
        regional_guidance: [],
      },
    },
    snapshots: [],
    stagingArea: { pendingImages: [] },
  },
  id,
  name,
  widgetInstances:
    selected === null
      ? {}
      : {
          gallery: {
            createdAt: '2026-01-01T00:00:00.000Z',
            id: 'gallery',
            state: { id: 'gallery', label: 'gallery', values: { selectedImageName: selected }, version: 1 },
            typeId: 'gallery',
          },
        },
  workflows: { activeWorkflowId: null, entries: [] },
});

type TestProject = ReturnType<typeof project>;

const usageDocument = (
  kind: GalleryMediaUsageDocument['kind'],
  id: string,
  name: string | null,
  imageNames: string[],
  imageCount = imageNames.length
): GalleryMediaUsageDocument => ({ id, imageCount, imageNames, kind, name, videoCount: 0, videoNames: [] });

const answer = (overrides: Partial<GalleryMediaUsage> = {}): GalleryMediaUsage => {
  const documents = overrides.documents ?? [];
  const names = new Set(documents.flatMap((entry) => entry.imageNames));

  return {
    checkedImageCount: 2,
    checkedVideoCount: 0,
    documents,
    documentsTotal: documents.length,
    inUseImageCount: names.size,
    inUseVideoCount: 0,
    otherAccounts: { documentCount: 0, imageCount: 0, queueItemCount: 0, videoCount: 0 },
    queue: { imageCount: 0, imageNames: [], itemCount: 0, videoCount: 0, videoNames: [] },
    ...overrides,
  };
};

/** Closed projects the server can return, by id. */
const serveProjects = (projects: readonly TestProject[]) => {
  mocks.getProject.mockImplementation((projectId) => {
    const found = projects.find((candidate) => candidate.id === projectId);

    return found ? Promise.resolve({ project: found }) : Promise.reject(new ApiError('Project not found', 404));
  });
};

const host = (loaded: readonly TestProject[] = [], openProjectId: string | null = null): MediaUsageHost => ({
  getLoadedProjects: () => loaded as unknown as ReturnType<MediaUsageHost['getLoadedProjects']>,
  openProjectId,
});

const items = (...names: string[]) => ({ kind: 'items' as const, refs: names.map(image) });

beforeEach(() => {
  accountLifecycle.activate('user-a');
  mocks.check.mockReset();
  mocks.getProject.mockReset();
});

afterEach(() => {
  accountLifecycle.invalidate();
});

describe('resolveMediaUsage', () => {
  it('drops a project the server names only for its gallery selection, and recounts what is in use', async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('project', 'p-a', 'Atlas', ['a.png']),
          usageDocument('project', 'p-b', 'Birch', ['b.png']),
        ],
      })
    );
    serveProjects([project('p-a', 'Atlas', { raster: ['a.png'] }), project('p-b', 'Birch', { selected: 'b.png' })]);

    const report = await resolveMediaUsage(items('a.png', 'b.png'), host());

    expect(report).toEqual({
      checkedCount: 2,
      groups: [{ count: 1, kind: 'project', name: 'Atlas', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
      // The server counted both; only one is used by work.
      inUseCount: 1,
      onlyImages: true,
      scope: 'items',
      serverCheck: 'complete',
    });
  });

  it('lists a project as the server counts it when it cannot be read, its names are capped, or past the read cap', async () => {
    const readable = Array.from({ length: MEDIA_USAGE_MAX_PROJECT_READS }, (_, index) => `p-${index}`);
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('project', 'p-capped', 'Capped', ['a.png'], 1500),
          ...readable.map((id) => usageDocument('project', id, id, ['a.png'])),
          usageDocument('project', 'p-late', 'Late', ['b.png']),
        ],
      })
    );
    // p-0 cannot be read; the others read fine but use nothing.
    serveProjects(readable.slice(1).map((id) => project(id, id)));

    const report = await resolveMediaUsage(items('a.png', 'b.png'), host());

    expect(mocks.getProject.mock.calls.map(([projectId]) => projectId).sort()).toEqual([...readable].sort());
    expect(report.groups).toEqual([
      { count: 1500, kind: 'projectUnchecked', name: 'Capped' },
      { count: 1, kind: 'projectUnchecked', name: 'p-0' },
      { count: 1, kind: 'projectUnchecked', name: 'Late' },
    ]);
    // The capped list's true count exceeds what was checked; the count never claims more than was asked about.
    expect(report.inUseCount).toBe(2);
  });

  it("trusts this tab's copy of a project over the server's, and never reads it", async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('project', 'open', 'Open', ['a.png']),
          usageDocument('project', 'tab-2', 'Second', ['b.png']),
        ],
      })
    );
    const loaded = [
      // Saved with a.png; since then the layer was replaced by an unsaved b.png.
      project('open', 'Open', { raster: ['b.png'] }),
      project('tab-2', 'Second', { raster: ['a.png'] }),
    ];

    const report = await resolveMediaUsage(items('a.png', 'b.png'), host(loaded, 'open'));

    expect(mocks.getProject).not.toHaveBeenCalled();
    expect(report.groups).toEqual([
      { count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
      { count: 1, kind: 'project', name: 'Second', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
    ]);
    expect(report.inUseCount).toBe(2);
  });

  it('narrows a board by the names the server found in each project', async () => {
    mocks.check.mockResolvedValue(
      answer({
        checkedImageCount: 5,
        checkedVideoCount: 1,
        documents: [usageDocument('project', 'open', 'Open', ['a.png', 'b.png'])],
      })
    );

    const report = await resolveMediaUsage(
      { boardId: 'board-1', kind: 'board' },
      host([project('open', 'Open', { raster: ['b.png'], selected: 'a.png' })], 'open')
    );

    expect(mocks.check).toHaveBeenCalledWith({ boardId: 'board-1', kind: 'board' }, expect.any(AbortSignal));
    expect(report).toMatchObject({
      checkedCount: 6,
      groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
      inUseCount: 1,
      onlyImages: false,
      scope: 'board',
    });
  });

  it("does not keep the server's count for a name a scan dropped, even when the scan found a new one", async () => {
    // The server saw a.png in the open project's recents; the project now uses c.png, not yet saved. Another account
    // uses c.png too, so the count cannot be read off the names alone.
    mocks.check.mockResolvedValue(
      answer({
        documents: [usageDocument('project', 'open', 'Open', ['a.png'])],
        inUseImageCount: 2,
        otherAccounts: { documentCount: 1, imageCount: 1, queueItemCount: 0, videoCount: 0 },
      })
    );

    const report = await resolveMediaUsage(
      items('a.png', 'c.png'),
      host([project('open', 'Open', { raster: ['c.png'], selected: 'a.png' })], 'open')
    );

    expect(report.inUseCount).toBe(1);
  });

  it('reads no closed project past the size bound, listing it as the server counts it', async () => {
    mocks.check.mockResolvedValue(answer({ documents: [usageDocument('project', 'p-big', 'Big', ['a.png'])] }));
    // What the transport answers for a record over the bound.
    mocks.getProject.mockResolvedValue(null);

    const report = await resolveMediaUsage(items('a.png'), host());

    expect(mocks.getProject).toHaveBeenCalledWith('p-big', MEDIA_USAGE_MAX_PROJECT_READ_BYTES, expect.any(AbortSignal));
    expect(report.groups).toEqual([{ count: 1, kind: 'projectUnchecked', name: 'Big' }]);
  });

  it('says a selection past what one check takes was too large, not that the server failed', async () => {
    const names = Array.from({ length: MEDIA_USAGE_MAX_NAMES_PER_KIND + 1 }, (_, index) => `${index}.png`);

    const report = await resolveMediaUsage(
      items(...names),
      host([project('open', 'Open', { raster: ['7.png'] })], 'open')
    );

    expect(mocks.check).not.toHaveBeenCalled();
    expect(report.serverCheck).toBe('tooManyItems');
    expect(report.groups).toEqual([
      { count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
    ]);
  });

  it('takes workflows, the legacy editor, projects kept for repair, queued jobs and other accounts as they are', async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('legacyEditor', 'legacy_editor', null, ['a.png']),
          usageDocument('quarantinedProject', 'q-1', 'Broken', ['b.png']),
          usageDocument('workflow', 'wf-1', 'Upscale pass', ['a.png', 'b.png']),
        ],
        documentsTotal: 105,
        otherAccounts: { documentCount: 1, imageCount: 1, queueItemCount: 1, videoCount: 0 },
        queue: { imageCount: 1, imageNames: ['a.png'], itemCount: 2, videoCount: 0, videoNames: [] },
      })
    );

    const report = await resolveMediaUsage(items('a.png', 'b.png'), host());

    expect(report.groups).toEqual([
      { count: 1, kind: 'quarantinedProject', name: 'Broken' },
      { count: 2, kind: 'workflow', name: 'Upscale pass' },
      { count: 1, kind: 'legacyEditor' },
      { documentCount: 102, kind: 'moreDocuments' },
      { count: 1, jobCount: 2, kind: 'queue' },
      { count: 1, kind: 'otherAccounts' },
    ]);
    expect(report.inUseCount).toBe(2);
  });

  it('still scans the projects this tab holds when the server check fails, and says the rest went unchecked', async () => {
    mocks.check.mockRejectedValue(new ApiError('Not Found', 404));

    const report = await resolveMediaUsage(
      items('a.png'),
      host([project('open', 'Open', { raster: ['a.png'] })], 'open')
    );

    expect(report).toEqual({
      checkedCount: 1,
      groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
      inUseCount: 1,
      onlyImages: true,
      scope: 'items',
      serverCheck: 'unavailable',
    });
    // A board's items are known only to the server.
    await expect(resolveMediaUsage({ boardId: 'board-1', kind: 'board' }, host())).rejects.toThrow();
  });

  it('gives up on a server that does not answer in time', async () => {
    vi.useFakeTimers();
    onTestFinished(() => {
      vi.useRealTimers();
    });
    mocks.check.mockImplementation(
      (_request, signal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason));
        })
    );

    const pending = resolveMediaUsage(items('a.png'), host(), { serverTimeoutMs: 10 });
    await vi.advanceTimersByTimeAsync(9);

    expect(mocks.check.mock.calls[0]?.[1].aborted).toBe(false);

    await vi.advanceTimersByTimeAsync(1);
    const report = await pending;

    expect(report.serverCheck).toBe('unavailable');
    expect(report.groups).toEqual([]);
  });

  it('lists a closed project as the server counts it when reading it outlasts the deadline', async () => {
    vi.useFakeTimers();
    onTestFinished(() => {
      vi.useRealTimers();
    });
    mocks.check.mockResolvedValue(answer({ documents: [usageDocument('project', 'p-slow', 'Slow', ['a.png'])] }));
    mocks.getProject.mockImplementation(
      (_projectId, _maxBytes, signal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason));
        })
    );

    const pending = resolveMediaUsage(items('a.png'), host(), { projectReadsTimeoutMs: 50 });
    await vi.advanceTimersByTimeAsync(50);
    const report = await pending;

    expect(report.groups).toEqual([{ count: 1, kind: 'projectUnchecked', name: 'Slow' }]);
  });

  it('stops without reading projects when the account changes during the check', async () => {
    let settle: (value: GalleryMediaUsage) => void = () => {};
    mocks.check.mockReturnValue(
      new Promise((resolve) => {
        settle = resolve;
      })
    );

    const pending = resolveMediaUsage(items('a.png'), host());

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    settle(answer({ documents: [usageDocument('project', 'p-a', 'Atlas', ['a.png'])] }));

    await expect(pending).rejects.toThrow();
    expect(mocks.getProject).not.toHaveBeenCalled();
  });
});
