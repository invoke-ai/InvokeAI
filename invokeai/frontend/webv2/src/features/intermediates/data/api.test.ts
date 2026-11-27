import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createIntermediatesPreview,
  getIntermediatesSummary,
  listIntermediatesOperations,
  mapIntermediatesOperation,
  startIntermediatesOperation,
} from './api';

const summaryDto = {
  can_manage_everyone: false,
  items: [
    {
      cover_image_name: null,
      images: { active: 1, recent: 2, referenced: 3, safe: 4 },
      project_id: null,
      project_name: null,
      reclaimable_bytes: 42,
      referenced_bytes: 7,
      unknown_size_count: 1,
      user_display_name: null,
      user_email: 'a@example.com',
      user_id: 'a',
      videos: { active: 0, recent: 0, referenced: 0, safe: 1 },
    },
  ],
  limit: 50,
  measuring: true,
  offset: 0,
  recent_grace_seconds: 1800,
  total: 1,
  totals: {
    in_use_images: 6,
    in_use_videos: 0,
    reclaimable_bytes: 42,
    rows: 1,
    safe_images: 4,
    safe_videos: 1,
    unknown_size_count: 1,
  },
};

const previewDto = (scope: unknown) => ({
  affected_documents: [],
  affected_documents_total: 0,
  created_at: 'now',
  expires_at: 'later',
  impact: {
    delete_images: 1,
    delete_videos: 0,
    keep_active_images: 0,
    keep_active_videos: 0,
    keep_recent_images: 0,
    keep_recent_videos: 0,
    keep_referenced_images: 0,
    keep_referenced_videos: 0,
    reclaimable_bytes: 5,
    unknown_size_count: 0,
  },
  mode: 'safe',
  preview_id: 'p1',
  scope,
  target_rows: 1,
});

const operationDto = {
  completed_at: null,
  created_at: 'now',
  error: null,
  mode: 'force' as const,
  operation_id: 'op',
  progress: {
    deleted_images: 1,
    deleted_videos: 0,
    failed_images: 2,
    failed_videos: 0,
    pending_disk_cleanup: 1,
    processed_images: 3,
    processed_videos: 0,
    reclaimed_bytes: 10,
    retained_images: 0,
    retained_videos: 0,
    unknown_size_count: 1,
  },
  scope: { kind: 'everyone' as const, targets: [], user_id: null },
  started_at: 'now',
  status: 'running' as const,
  target_images: 3,
  target_videos: 0,
  user_id: 'admin',
};

const fetchJson = (body: unknown, status = 200) =>
  vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
    Promise.resolve(new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' }, status }))
  );

describe('intermediates transport', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps the summary to camelCase read models and only sends set filters', async () => {
    const fetchMock = fetchJson(summaryDto);
    vi.stubGlobal('fetch', fetchMock);

    const summary = await getIntermediatesSummary({ limit: 50, ownerId: null, search: '  ', sort: 'project_name' });

    expect(String(fetchMock.mock.calls[0]?.[0])).toMatch(
      /\/api\/v1\/intermediates\/summary\?sort=project_name&limit=50$/
    );
    expect(summary.items[0]).toMatchObject({
      images: { active: 1, recent: 2, referenced: 3, safe: 4 },
      projectId: null,
      reclaimableBytes: 42,
      referencedBytes: 7,
      unknownSizeCount: 1,
      userId: 'a',
    });
    expect(summary.totals).toEqual({
      inUseImages: 6,
      inUseVideos: 0,
      reclaimableBytes: 42,
      rows: 1,
      safeImages: 4,
      safeVideos: 1,
      unknownSizeCount: 1,
    });
    expect(summary.measuring).toBe(true);
    expect(summary.recentGraceSeconds).toBe(1800);
  });

  it('serializes a selection scope with snake_case targets on the flat wire model', async () => {
    const fetchMock = fetchJson(
      previewDto({ kind: 'selection', targets: [{ project_id: null, user_id: 'a' }], user_id: null }),
      201
    );
    vi.stubGlobal('fetch', fetchMock);

    const preview = await createIntermediatesPreview({
      mode: 'safe',
      scope: { kind: 'selection', targets: [{ projectId: null, userId: 'a' }] },
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(String(init.body))).toEqual({
      mode: 'safe',
      scope: {
        excluded: [],
        kind: 'selection',
        project_id: null,
        search: null,
        targets: [{ project_id: null, user_id: 'a' }],
        user_id: null,
      },
    });
    expect(preview.scope).toEqual({ kind: 'selection', targets: [{ projectId: null, userId: 'a' }] });
    expect(preview.impact.deleteImages).toBe(1);
  });

  it('round-trips a matching scope with its filters and exclusions', async () => {
    const wire = {
      excluded: [{ project_id: null, user_id: 'a' }],
      kind: 'matching',
      project_id: 'p',
      search: 'port',
      targets: [],
      user_id: null,
    };
    const fetchMock = fetchJson(previewDto(wire), 201);
    vi.stubGlobal('fetch', fetchMock);
    const scope = {
      excluded: [{ projectId: null, userId: 'a' }],
      kind: 'matching' as const,
      projectId: 'p',
      search: 'port',
      userId: null,
    };

    const preview = await createIntermediatesPreview({ mode: 'safe', scope });

    expect(JSON.parse(String((fetchMock.mock.calls[0]![1] as RequestInit).body)).scope).toEqual(wire);
    expect(preview.scope).toEqual(scope);
  });

  it('maps operation progress and scope', () => {
    const operation = mapIntermediatesOperation(operationDto);

    expect(operation.progress).toMatchObject({ failedImages: 2, pendingDiskCleanup: 1, unknownSizeCount: 1 });
    expect(operation.scope).toEqual({ kind: 'everyone' });
    expect(operation.targetImages).toBe(3);
  });

  it('starts with only the preview id and lists operations from the items envelope', async () => {
    const startFetch = fetchJson(operationDto, 202);
    vi.stubGlobal('fetch', startFetch);
    const started = await startIntermediatesOperation({ previewId: 'p1' }, new AbortController().signal);
    expect(JSON.parse(String((startFetch.mock.calls[0]![1] as RequestInit).body))).toEqual({ preview_id: 'p1' });
    expect(started.operationId).toBe('op');

    const listFetch = fetchJson({ items: [operationDto, { ...operationDto, operation_id: 'older' }] });
    vi.stubGlobal('fetch', listFetch);
    const listed = await listIntermediatesOperations();
    expect(String(listFetch.mock.calls[0]?.[0])).toMatch(/\/api\/v1\/intermediates\/operations$/);
    expect(listed.map((operation) => operation.operationId)).toEqual(['op', 'older']);
  });
});
