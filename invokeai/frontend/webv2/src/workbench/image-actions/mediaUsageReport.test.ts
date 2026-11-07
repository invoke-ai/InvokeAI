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
  type MediaUsageSyncBase,
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

const widget = (typeId: string, values: Record<string, unknown>) => ({
  createdAt: '2026-01-01T00:00:00.000Z',
  id: typeId,
  state: { id: typeId, label: typeId, values, version: 1 },
  typeId,
});

/**
 * A project whose canvas raster layers hold `raster`; `selected` is only the Gallery's selection, `pinned` a name in a
 * widget the usage panel has no surface for.
 */
const project = (
  id: string,
  name: string,
  { pinned = null as string | null, raster = [] as string[], selected = null as string | null } = {}
) => ({
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
  widgetInstances: {
    ...(selected === null ? {} : { gallery: widget('gallery', { selectedImageName: selected }) }),
    ...(pinned === null ? {} : { 'image-notes': widget('image-notes', { pinned: { imageName: pinned } }) }),
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
): GalleryMediaUsageDocument => ({
  id,
  imageCount,
  imageNames,
  kind,
  name,
  revision: null,
  videoCount: 0,
  videoNames: [],
  workspaceRevision: null,
});

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

/** The server's copies of projects, by id, each at revision 4 of its content and 2 of the caller's workspace. */
const serveProjects = (projects: readonly TestProject[]) => {
  mocks.getProject.mockImplementation((projectId) => {
    const found = projects.find((candidate) => candidate.id === projectId);

    return found
      ? Promise.resolve({ project: found, revision: 4, workspace: { revision: 2 } })
      : Promise.reject(new ApiError('Project not found', 404));
  });
};

/** What the server's copies in `serveProjects` build on. */
const SERVED: MediaUsageSyncBase = { revision: 4, workspaceRevision: 2 };

/** A project entry as the server reads it at the revisions `serveProjects` serves. */
const atServed = (entry: GalleryMediaUsageDocument): GalleryMediaUsageDocument => ({
  ...entry,
  revision: 4,
  workspaceRevision: 2,
});

/** This tab's projects, and by id the server revisions each copy builds on (unknown when absent). */
const host = (
  loaded: readonly TestProject[] = [],
  openProjectId: string | null = null,
  bases: Readonly<Record<string, MediaUsageSyncBase>> = {}
): MediaUsageHost => ({
  getLoadedProjects: () => loaded as unknown as ReturnType<MediaUsageHost['getLoadedProjects']>,
  getSyncBase: (projectId) => bases[projectId],
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

  it('keeps a project whose only use of the media is a field the panel has no surface for', async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('project', 'p-a', 'Atlas', ['a.png']),
          usageDocument('project', 'p-b', 'Birch', ['b.png']),
        ],
      })
    );
    serveProjects([project('p-a', 'Atlas', { pinned: 'a.png' })]);

    const report = await resolveMediaUsage(
      items('a.png', 'b.png'),
      host([project('p-b', 'Birch', { pinned: 'b.png' })])
    );

    expect(report.groups).toEqual([
      { count: 1, kind: 'project', name: 'Birch', surfaces: [{ count: 1, surface: 'otherProjectState' }] },
      { count: 1, kind: 'project', name: 'Atlas', surfaces: [{ count: 1, surface: 'otherProjectState' }] },
    ]);
    expect(report.inUseCount).toBe(2);
  });

  it('counts the items asked about, as the dialog title does, even when the server could check fewer', async () => {
    // Of three selected items on a shared board the caller owns one: the server checks only that one, without saying
    // which, so the projects here are scanned for all three.
    mocks.check.mockResolvedValue(answer({ checkedImageCount: 1 }));

    const report = await resolveMediaUsage(
      items('mine.png', 'theirs-1.png', 'theirs-2.png'),
      host([project('open', 'Open', { raster: ['theirs-1.png'] })], 'open')
    );

    expect(report).toMatchObject({ checkedCount: 3, inUseCount: 1 });
  });

  it('tells an image from a video of the same name when counting what a project uses', async () => {
    mocks.check.mockRejectedValue(new ApiError('Not Found', 404));
    const loaded = {
      ...project('open', 'Open'),
      widgetInstances: { video: widget('video', { sourceVideo: { video_name: 'same' } }) },
    };

    const report = await resolveMediaUsage(
      { kind: 'items', refs: [image('same'), { kind: 'video', name: 'same' }] },
      host([loaded as unknown as TestProject], 'open')
    );

    expect(report.groups).toEqual([{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'video' }] }]);
    expect(report.inUseCount).toBe(1);
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

  it("trusts this tab's copy without reading the server's when the server holds nothing newer", async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          atServed(usageDocument('project', 'open', 'Open', ['a.png'])),
          // What this copy builds on is unknown, but its scan reports every name the server found.
          usageDocument('project', 'tab-2', 'Second', ['b.png']),
        ],
      })
    );
    const loaded = [
      // Saved with a.png; since then the layer was replaced by an unsaved b.png.
      project('open', 'Open', { raster: ['b.png'] }),
      // Saved with b.png; a.png added since, unsaved.
      project('tab-2', 'Second', { raster: ['a.png', 'b.png'] }),
    ];

    const report = await resolveMediaUsage(items('a.png', 'b.png'), host(loaded, 'open', { open: SERVED }));

    expect(mocks.getProject).not.toHaveBeenCalled();
    expect(report.groups).toEqual([
      { count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
      { count: 2, kind: 'project', name: 'Second', surfaces: [{ count: 2, surface: 'canvasRaster' }] },
    ]);
    expect(report.inUseCount).toBe(2);
  });

  it('drops a loaded project the server names only for its Gallery selection, without reading it', async () => {
    // The common deletion: the image the user is looking at, saved as the selection and nothing else.
    mocks.check.mockResolvedValue(
      answer({ documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png']))] })
    );

    const report = await resolveMediaUsage(
      items('a.png'),
      host([project('open', 'Open', { selected: 'a.png' })], 'open', { open: SERVED })
    );

    expect(mocks.getProject).not.toHaveBeenCalled();
    expect(report).toMatchObject({ groups: [], inUseCount: 0, serverCheck: 'complete' });
  });

  it('narrows a board by the names the server found in each project', async () => {
    mocks.check.mockResolvedValue(
      answer({
        checkedImageCount: 5,
        checkedVideoCount: 1,
        documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png', 'b.png']))],
      })
    );

    const report = await resolveMediaUsage(
      { boardId: 'board-1', kind: 'board' },
      host([project('open', 'Open', { raster: ['b.png'], selected: 'a.png' })], 'open', { open: SERVED })
    );

    expect(mocks.getProject).not.toHaveBeenCalled();
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
        documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png']))],
        inUseImageCount: 2,
        otherAccounts: { documentCount: 1, imageCount: 1, queueItemCount: 0, videoCount: 0 },
      })
    );

    const report = await resolveMediaUsage(
      items('a.png', 'c.png'),
      host([project('open', 'Open', { raster: ['c.png'], selected: 'a.png' })], 'open', { open: SERVED })
    );

    expect(report.inUseCount).toBe(1);
  });

  it.each([
    ['its content is behind', { revision: 3, workspaceRevision: 2 }],
    ['its workspace is behind', { revision: 4, workspaceRevision: 1 }],
    ['what it builds on is unknown', undefined],
  ] as const)(
    "reads the server's copy and keeps a use it holds when this tab's copy may be older: %s",
    async (_case, base) => {
      // Another tab or device pinned a.png and saved; this tab's copy, unsaved since, uses only b.png.
      mocks.check.mockResolvedValue(
        answer({ documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png', 'b.png']))] })
      );
      serveProjects([project('open', 'Open', { pinned: 'a.png', raster: ['b.png'] })]);

      const report = await resolveMediaUsage(
        items('a.png', 'b.png'),
        host([project('open', 'Open', { raster: ['b.png'] })], 'open', base ? { open: base } : {})
      );

      expect(mocks.getProject.mock.calls.map(([projectId]) => projectId)).toEqual(['open']);
      expect(report.groups).toEqual([
        {
          count: 2,
          kind: 'openProject',
          surfaces: [
            { count: 1, surface: 'canvasRaster' },
            { count: 1, surface: 'otherProjectState' },
          ],
        },
      ]);
      expect(report.inUseCount).toBe(2);
    }
  );

  it("reads the server's copy when the server did not say which revisions it read, and trusts this tab's if it builds on it", async () => {
    mocks.check.mockResolvedValue(answer({ documents: [usageDocument('project', 'open', 'Open', ['a.png'])] }));
    serveProjects([project('open', 'Open', { raster: ['a.png'] })]);

    const report = await resolveMediaUsage(
      items('a.png', 'b.png'),
      host([project('open', 'Open', { raster: ['b.png'] })], 'open', { open: SERVED })
    );

    expect(mocks.getProject.mock.calls.map(([projectId]) => projectId)).toEqual(['open']);
    expect(report.groups).toEqual([
      { count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
    ]);
  });

  it("keeps the server's count for an open project whose saved version cannot be read, beside what its copy uses", async () => {
    mocks.check.mockResolvedValue(
      answer({ documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png']))] })
    );
    mocks.getProject.mockRejectedValue(new ApiError('Service Unavailable', 503));

    const report = await resolveMediaUsage(
      items('a.png', 'b.png'),
      host([project('open', 'Open', { raster: ['b.png'] })], 'open', { open: { revision: 3, workspaceRevision: 2 } })
    );

    // The server's a.png stands beside the b.png found here.
    expect(report.groups).toEqual([
      { count: 2, kind: 'openProject', savedCopyUnchecked: true, surfaces: [{ count: 1, surface: 'canvasRaster' }] },
    ]);
    expect(report.inUseCount).toBe(2);
  });

  it.each([
    ['in conflict', { conflict: { detectedAt: '2026-01-01T00:00:00.000Z', kind: 'revision' as const } }],
    [
      'refused for its schema',
      { schemaRefusal: { kind: 'canvas' as const, maxCanvasSchemaVersion: 3, minimumCanvasSchemaVersion: 4 } },
    ],
  ])(
    "reads and joins the server's copy of a project %s, whose reported revisions its copy does not build on",
    async (_case, decision) => {
      // Saved from another device with a.png at the reported revisions; this tab shows a diverged draft without it.
      mocks.check.mockResolvedValue(
        answer({ documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png']))] })
      );
      serveProjects([project('open', 'Open', { raster: ['a.png'] })]);

      const report = await resolveMediaUsage(
        items('a.png'),
        host([project('open', 'Open')], 'open', { open: { ...SERVED, ...decision } })
      );

      expect(mocks.getProject.mock.calls.map(([projectId]) => projectId)).toEqual(['open']);
      expect(report.groups).toEqual([
        { count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] },
      ]);
    }
  );

  it('reads a project open here that needs it before closed projects, within the read cap', async () => {
    const closed = Array.from({ length: MEDIA_USAGE_MAX_PROJECT_READS }, (_, index) => `p-${index}`);
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          ...closed.map((id) => usageDocument('project', id, id, ['a.png'])),
          usageDocument('project', 'open', 'Open', ['a.png']),
        ],
      })
    );
    serveProjects([project('open', 'Open'), ...closed.map((id) => project(id, id))]);

    await resolveMediaUsage(items('a.png'), host([project('open', 'Open')], 'open'));

    expect(mocks.getProject.mock.calls.map(([projectId]) => projectId).sort()).toEqual(
      ['open', ...closed.slice(0, -1)].sort()
    );
  });

  it("keeps a board item the server's newer copy uses, listed once beside the uses found here", async () => {
    mocks.check.mockResolvedValue(
      answer({
        checkedImageCount: 3,
        documents: [atServed(usageDocument('project', 'open', 'Open', ['a.png', 'b.png']))],
      })
    );
    // Saved elsewhere: a.png on a layer and pinned. Here, unsaved: the a.png layer replaced by b.png.
    serveProjects([project('open', 'Open', { pinned: 'a.png', raster: ['a.png'] })]);

    const report = await resolveMediaUsage(
      { boardId: 'board-1', kind: 'board' },
      host([project('open', 'Open', { raster: ['b.png'] })], 'open', { open: { revision: 3, workspaceRevision: 2 } })
    );

    expect(report).toMatchObject({
      groups: [{ count: 2, kind: 'openProject', surfaces: [{ count: 2, surface: 'canvasRaster' }] }],
      inUseCount: 2,
    });
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

  it('takes workflows, the legacy editor, projects kept for repair, unknown documents, queued jobs and other accounts as they are', async () => {
    mocks.check.mockResolvedValue(
      answer({
        documents: [
          usageDocument('legacyEditor', 'legacy_editor', null, ['a.png']),
          usageDocument('quarantinedProject', 'q-1', 'Broken', ['b.png']),
          usageDocument('workflow', 'wf-1', 'Upscale pass', ['a.png', 'b.png']),
          // A kind of saved document this build does not know.
          usageDocument('other', 'board-notes-1', 'Board notes', ['b.png']),
          usageDocument('other', 'unnamed-1', null, ['a.png']),
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
      { count: 1, kind: 'otherDocument', name: 'Board notes' },
      { count: 1, kind: 'otherDocument', name: null },
      { documentCount: 100, kind: 'moreDocuments' },
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
