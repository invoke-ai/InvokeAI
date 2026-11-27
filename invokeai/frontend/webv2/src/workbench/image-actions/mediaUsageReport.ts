/**
 * Where media about to be deleted is still used: the server's reference check narrowed by each named project's scan
 * (`collectMediaUsage`), which fails safe by reporting any name outside fields known only to show or record media. A
 * loaded copy's scan stands only when it covers every name the server found or builds on the exact revisions the server
 * read; otherwise the server's copy is read and a ref either copy uses counts. Only the server knows board members.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { Project } from '@workbench/projectContracts';
import type { ProjectSyncInfo } from '@workbench/projects/syncStore';

import { toGalleryItemKey } from '@features/gallery/contracts';
import {
  checkGalleryMediaUsage,
  GALLERY_REQUEST_MAX_NAMES,
  type GalleryMediaUsage,
  type GalleryMediaUsageDocument,
  type GalleryMediaUsageGroup,
  type GalleryMediaUsageReport,
  type GalleryMediaUsageRequest,
  type GalleryMediaUsageSurfaceRow,
} from '@features/gallery/mediaUsage';
import { captureAccountScope, assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getProjectWithinSize } from '@workbench/projects/api';
import { deserializeProjectRecord } from '@workbench/projects/projectHydration';

import { collectMediaUsage, mergeMediaUsage, type MediaUsage } from './mediaUsage';

/** Each read transfers a whole project document; past the cap a project is listed as the server counts it. */
export const MEDIA_USAGE_MAX_PROJECT_READS = 6;
/** Larger projects are not read: parsing and migrating would hold the main thread while the user waits on a dialog. */
export const MEDIA_USAGE_MAX_PROJECT_READ_BYTES = 4 * 1024 * 1024;
const PROJECT_READ_CONCURRENCY = 2;
const DEFAULT_SERVER_TIMEOUT_MS = 10_000;
const DEFAULT_PROJECT_READS_TIMEOUT_MS = 5_000;

/** A copy in conflict or refused holds local work the reported revisions do not describe. */
export type MediaUsageSyncBase = Pick<ProjectSyncInfo, 'conflict' | 'revision' | 'schemaRefusal' | 'workspaceRevision'>;

export interface MediaUsageHost {
  getLoadedProjects(): readonly Project[];
  /** Undefined, or a null revision, when unknown (never saved, no persistence). */
  getSyncBase(projectId: string): MediaUsageSyncBase | undefined;
  openProjectId: string | null;
}

export interface MediaUsageOptions {
  /** Per request: a selection past one request's cap is checked in several. */
  serverTimeoutMs?: number;
  /** Projects not read by then are listed as the server counts them. */
  projectReadsTimeoutMs?: number;
  signal?: AbortSignal;
}

/**
 * A plain timer, not `AbortSignal.timeout`: a timeout signal reachable only through `AbortSignal.any` can be garbage
 * collected before it fires, and the stage would wait forever.
 */
const withDeadline = (signal: AbortSignal, ms: number): { clear: () => void; signal: AbortSignal } => {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal.reason);
  const timer = setTimeout(() => controller.abort(new DOMException('The usage check timed out.', 'TimeoutError')), ms);

  if (signal.aborted) {
    onAbort();
  } else {
    signal.addEventListener('abort', onAbort, { once: true });
  }

  return {
    clear: () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    },
    signal: controller.signal,
  };
};

const isTruncated = (entry: {
  imageCount: number;
  imageNames: readonly string[];
  namesIncomplete?: true;
  videoCount: number;
  videoNames: readonly string[];
}): boolean =>
  entry.namesIncomplete === true ||
  entry.imageCount > entry.imageNames.length ||
  entry.videoCount > entry.videoNames.length;

const countBoundOf = (entry: GalleryMediaUsageDocument): { countAtLeast?: true } =>
  entry.namesIncomplete ? { countAtLeast: true } : {};

const refsOf = (entry: GalleryMediaUsageDocument): GalleryItemRef[] => [
  ...entry.imageNames.map((name) => ({ kind: 'image' as const, name })),
  ...entry.videoNames.map((name) => ({ kind: 'video' as const, name })),
];

const dedupeRefs = (refs: readonly GalleryItemRef[]): GalleryItemRef[] => {
  const seen = new Set<string>();

  return refs.filter((ref) => {
    const key = toGalleryItemKey(ref);

    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
};

const dropsServerNames = (entry: GalleryMediaUsageDocument, used: ReadonlySet<string>): boolean =>
  // A capped list hides names the scan may have dropped; more counted than the scan kept means some were.
  refsOf(entry).some((ref) => !used.has(toGalleryItemKey(ref))) || entry.imageCount + entry.videoCount > used.size;

interface ServerProject {
  project: Project;
  revision: number;
  workspaceRevision: number | null;
}

const readServerProject = async (projectId: string, signal: AbortSignal): Promise<ServerProject | null> => {
  const record = await getProjectWithinSize(projectId, MEDIA_USAGE_MAX_PROJECT_READ_BYTES, signal);
  const loaded = record ? deserializeProjectRecord(record) : null;

  return record && loaded?.status === 'loaded'
    ? { project: loaded.project, revision: record.revision, workspaceRevision: record.workspace?.revision ?? null }
    : null;
};

/** Unknown revisions on either side never match, nor does a copy in conflict or refused. */
const isServerCopyOf = (
  base: MediaUsageSyncBase | undefined,
  server: { revision: number | null; workspaceRevision: number | null }
): boolean =>
  base !== undefined &&
  base.conflict === undefined &&
  base.schemaRefusal === undefined &&
  base.revision !== null &&
  base.workspaceRevision !== undefined &&
  server.revision !== null &&
  base.revision === server.revision &&
  base.workspaceRevision === server.workspaceRevision;

const runLimited = async <Item>(
  items: readonly Item[],
  limit: number,
  worker: (item: Item) => Promise<void>
): Promise<void> => {
  let next = 0;

  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const item = items[next];
        next += 1;
        await worker(item!);
      }
    })
  );
};

const usedKeysOf = (usage: readonly MediaUsage[]): Set<string> =>
  new Set(usage.flatMap((entry) => entry.refs.map(toGalleryItemKey)));

const toSurfaceRows = (usage: readonly MediaUsage[]): GalleryMediaUsageSurfaceRow[] =>
  usage.map(({ label, refs, surface }) =>
    label === undefined ? { count: refs.length, surface } : { count: refs.length, label, surface }
  );

type ProjectOutcome =
  | { kind: 'scanned'; usage: MediaUsage[] }
  | { kind: 'unchecked'; entry: GalleryMediaUsageDocument; local?: MediaUsage[] };

interface ProjectRead {
  entry: GalleryMediaUsageDocument;
  local?: { base: MediaUsageSyncBase | undefined; usage: MediaUsage[] };
  refs: GalleryItemRef[];
}

/** For items a failed server check still scans loaded projects and marks the rest unchecked; for a board it rejects. */
export const resolveMediaUsage = async (
  request: GalleryMediaUsageRequest,
  host: MediaUsageHost,
  {
    projectReadsTimeoutMs = DEFAULT_PROJECT_READS_TIMEOUT_MS,
    serverTimeoutMs = DEFAULT_SERVER_TIMEOUT_MS,
    signal: callerSignal,
  }: MediaUsageOptions = {}
): Promise<GalleryMediaUsageReport> => {
  const owner = captureAccountScope();
  const signal = callerSignal ? AbortSignal.any([callerSignal, owner.signal]) : owner.signal;
  const requestedRefs = request.kind === 'items' ? dedupeRefs(request.refs) : [];
  let server: GalleryMediaUsage | null = null;
  const serverRequests = Math.max(
    1,
    ...(['image', 'video'] as const).map((kind) =>
      Math.ceil(requestedRefs.filter((ref) => ref.kind === kind).length / GALLERY_REQUEST_MAX_NAMES)
    )
  );
  const deadline = withDeadline(signal, serverTimeoutMs * serverRequests);

  try {
    server = await checkGalleryMediaUsage(
      request.kind === 'items' ? { kind: 'items', refs: requestedRefs } : request,
      deadline.signal
    );
  } catch {
    server = null;
  } finally {
    deadline.clear();
  }

  signal.throwIfAborted();
  assertAccountScopeCurrent(owner);

  if (server === null && request.kind === 'board') {
    throw new Error('The media usage check did not answer for the board.');
  }

  const documents = server?.documents ?? [];
  const serverProjects = new Map(
    documents.filter((entry) => entry.kind === 'project').map((entry) => [entry.id, entry] as const)
  );
  const outcomes = new Map<string, ProjectOutcome>();
  const loadedProjects = host.getLoadedProjects();
  const reads: ProjectRead[] = [];

  for (const project of loadedProjects) {
    const entry = serverProjects.get(project.id);

    if (request.kind === 'board' && entry && isTruncated(entry)) {
      outcomes.set(project.id, { entry, kind: 'unchecked' });
      continue;
    }

    const refs = request.kind === 'items' ? requestedRefs : entry ? refsOf(entry) : [];
    const usage = collectMediaUsage(project, refs);
    const localKeys = usedKeysOf(usage);

    const base = host.getSyncBase(project.id);

    // A document a batched check could not name in full cannot vouch for a scan.
    if (entry && (entry.namesIncomplete || dropsServerNames(entry, localKeys)) && !isServerCopyOf(base, entry)) {
      // The server's copy decides: an edit here removed the name, or another device saved a use this copy lacks.
      outcomes.set(project.id, { entry, kind: 'unchecked', local: usage });
      reads.push({ entry, local: { base, usage }, refs });
    } else {
      outcomes.set(project.id, { kind: 'scanned', usage });
    }
  }

  const candidates = [...serverProjects.values()].filter((entry) => !outcomes.has(entry.id));

  for (const entry of candidates) {
    outcomes.set(entry.id, { entry, kind: 'unchecked' });
    if (!isTruncated(entry)) {
      reads.push({ entry, refs: refsOf(entry) });
    }
  }

  const readsDeadline = withDeadline(signal, projectReadsTimeoutMs);
  const readsSignal = readsDeadline.signal;

  await runLimited(reads.slice(0, MEDIA_USAGE_MAX_PROJECT_READS), PROJECT_READ_CONCURRENCY, async (read) => {
    try {
      readsSignal.throwIfAborted();
      const server = await readServerProject(read.entry.id, readsSignal);

      if (!server) {
        return;
      }

      const { local } = read;
      const usage = !local
        ? collectMediaUsage(server.project, read.refs)
        : isServerCopyOf(local.base, server)
          ? local.usage
          : mergeMediaUsage(local.usage, collectMediaUsage(server.project, read.refs));

      outcomes.set(read.entry.id, { kind: 'scanned', usage });
    } catch {
      // Listed as the server counts it.
    }
  });
  readsDeadline.clear();

  signal.throwIfAborted();
  assertAccountScopeCurrent(owner);

  return buildReport({ documents, host, loadedProjects, outcomes, request, requestedRefs, server });
};

const buildReport = ({
  documents,
  host,
  loadedProjects,
  outcomes,
  request,
  requestedRefs,
  server,
}: {
  documents: readonly GalleryMediaUsageDocument[];
  host: MediaUsageHost;
  loadedProjects: ReturnType<MediaUsageHost['getLoadedProjects']>;
  outcomes: ReadonlyMap<string, ProjectOutcome>;
  request: GalleryMediaUsageRequest;
  requestedRefs: readonly GalleryItemRef[];
  server: GalleryMediaUsage | null;
}): GalleryMediaUsageReport => {
  const groups: GalleryMediaUsageGroup[] = [];
  const inUse = new Set<string>();
  let lowerBound = 0;
  let exact = true;
  // A scan that dropped a server name invalidates the server's own in-use total.
  let narrowed = false;
  const addNames = (entry: Parameters<typeof isTruncated>[0]) => {
    entry.imageNames.forEach((name) => inUse.add(toGalleryItemKey({ kind: 'image', name })));
    entry.videoNames.forEach((name) => inUse.add(toGalleryItemKey({ kind: 'video', name })));
    if (isTruncated(entry)) {
      exact = false;
      lowerBound = Math.max(lowerBound, entry.imageCount + entry.videoCount);
    }
  };
  const serverEntryOf = (projectId: string) =>
    documents.find((document) => document.kind === 'project' && document.id === projectId);
  const projectGroup = (projectId: string, name: string | null): GalleryMediaUsageGroup | null => {
    const outcome = outcomes.get(projectId);

    if (!outcome) {
      return null;
    }
    if (outcome.kind === 'unchecked') {
      const { entry, local } = outcome;

      addNames(entry);
      if (!local) {
        return { count: entry.imageCount + entry.videoCount, kind: 'projectUnchecked', name, ...countBoundOf(entry) };
      }

      const localKeys = usedKeysOf(local);
      const count = Math.max(
        entry.imageCount + entry.videoCount,
        new Set([...refsOf(entry).map(toGalleryItemKey), ...localKeys]).size
      );
      const surfaces = toSurfaceRows(local);

      localKeys.forEach((key) => inUse.add(key));
      return projectId === host.openProjectId
        ? { count, kind: 'openProject', savedCopyUnchecked: true, surfaces, ...countBoundOf(entry) }
        : { count, kind: 'project', name, savedCopyUnchecked: true, surfaces, ...countBoundOf(entry) };
    }

    const used = usedKeysOf(outcome.usage);
    const entry = serverEntryOf(projectId);

    if (entry && dropsServerNames(entry, used)) {
      narrowed = true;
    }
    if (used.size === 0) {
      return null;
    }
    used.forEach((key) => inUse.add(key));

    const surfaces = toSurfaceRows(outcome.usage);

    return projectId === host.openProjectId
      ? { count: used.size, kind: 'openProject', surfaces }
      : { count: used.size, kind: 'project', name, surfaces };
  };
  const loadedIds = new Set(loadedProjects.map((project) => project.id));
  const openProject = loadedProjects.find((project) => project.id === host.openProjectId);
  const openGroup = openProject ? projectGroup(openProject.id, openProject.name) : null;

  if (openGroup) {
    groups.push(openGroup);
  }
  for (const project of loadedProjects) {
    const group = project.id === host.openProjectId ? null : projectGroup(project.id, project.name);

    if (group) {
      groups.push(group);
    }
  }
  for (const entry of documents) {
    if (entry.kind === 'project') {
      const group = loadedIds.has(entry.id) ? null : projectGroup(entry.id, entry.name);

      if (group) {
        groups.push(group);
      }
    }
  }

  const listed = (
    kind: GalleryMediaUsageDocument['kind'],
    toGroup: (entry: GalleryMediaUsageDocument) => GalleryMediaUsageGroup
  ) => {
    for (const entry of documents) {
      if (entry.kind === kind) {
        addNames(entry);
        groups.push({ ...toGroup(entry), ...countBoundOf(entry) });
      }
    }
  };
  const countOf = (entry: GalleryMediaUsageDocument) => entry.imageCount + entry.videoCount;

  listed('quarantinedProject', (entry) => ({ count: countOf(entry), kind: 'quarantinedProject', name: entry.name }));
  listed('workflow', (entry) => ({ count: countOf(entry), kind: 'workflow', name: entry.name }));
  listed('legacyEditor', (entry) => ({ count: countOf(entry), kind: 'legacyEditor' }));
  listed('other', (entry) => ({ count: countOf(entry), kind: 'otherDocument', name: entry.name }));

  if (server) {
    const unlisted = server.documentsTotal - server.documents.length;

    if (unlisted > 0) {
      exact = false;
      groups.push({ documentCount: unlisted, kind: 'moreDocuments' });
    }
    if (server.queue.itemCount > 0) {
      addNames(server.queue);
      groups.push({
        count: server.queue.imageCount + server.queue.videoCount,
        jobCount: server.queue.itemCount,
        kind: 'queue',
      });
    }

    const others = server.otherAccounts;

    if (others.documentCount + others.queueItemCount > 0) {
      exact = false;
      lowerBound = Math.max(lowerBound, others.imageCount + others.videoCount);
      groups.push({ count: others.imageCount + others.videoCount, kind: 'otherAccounts' });
    }
  }

  // The server's checked counts exclude items the caller may not delete without naming them, so count what was asked.
  const checkedCount =
    request.kind === 'items' || !server ? requestedRefs.length : server.checkedImageCount + server.checkedVideoCount;
  // Documents the server did not list count nowhere else, so with any of those the higher total stands.
  const serverInUse =
    server && (!narrowed || server.documentsTotal > server.documents.length)
      ? server.inUseImageCount + server.inUseVideoCount
      : 0;
  const inUseCount = exact ? inUse.size : Math.max(inUse.size, lowerBound, serverInUse);

  return {
    checkedCount,
    groups,
    inUseCount: Math.min(inUseCount, checkedCount),
    onlyImages:
      request.kind === 'items'
        ? requestedRefs.every((ref) => ref.kind === 'image')
        : server !== null && server.checkedVideoCount === 0,
    scope: request.kind,
    serverCheck: server ? 'complete' : 'unavailable',
  };
};

/** A last look before deleting without asking: a use added while the check ran is not in its answer. */
export const isUsedInLoadedProjects = (refs: readonly GalleryItemRef[], host: MediaUsageHost): boolean =>
  host.getLoadedProjects().some((project) => collectMediaUsage(project, refs).length > 0);
