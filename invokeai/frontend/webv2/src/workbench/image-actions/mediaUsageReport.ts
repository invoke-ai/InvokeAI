/**
 * Where media about to be deleted is still used, across the account: the server's reference check narrowed by the
 * projects' own scans. The server counts every media name a project document holds, including the Gallery's selection
 * and recents and workflow last runs, which only show or record media; the deletion breaks the rest. So each project
 * the server names is scanned here (`collectMediaUsage`), from memory when this tab has it loaded, else read from the
 * server without opening it in the editor, and is dropped only when the scan finds nothing. The scan fails safe: it
 * leaves out only fields it knows show or record media, and reports a name anywhere else in the project, so a field it
 * has no surface for keeps the project in the report. Workflows, the legacy editor, projects kept for repair, queued
 * jobs and other accounts are taken from the server as they are.
 *
 * Projects in memory are fresher than the server, which autosaves after a short debounce. For items that holds in
 * full: the requested refs are scanned. For a board it does not: only the server knows the board's members, so a
 * project in memory is scanned for the names the server attributed to it, and a use added since its last save is
 * missed.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { Project } from '@workbench/projectContracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import {
  checkGalleryMediaUsage,
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

import { collectMediaUsage, type MediaUsage } from './mediaUsage';

/** The server checks at most this many names of each kind per request. */
export const MEDIA_USAGE_MAX_NAMES_PER_KIND = 1000;
/**
 * Closed projects read to narrow the server's answer, at most, a few at a time. Each read transfers a whole project
 * document; past the cap a project is listed as the server counts it.
 */
export const MEDIA_USAGE_MAX_PROJECT_READS = 6;
/**
 * A closed project larger than this is not read: fetching, parsing and migrating it would hold the main thread while
 * the user waits on a dialog. It is listed as the server counts it.
 */
export const MEDIA_USAGE_MAX_PROJECT_READ_BYTES = 4 * 1024 * 1024;
const PROJECT_READ_CONCURRENCY = 2;
const DEFAULT_SERVER_TIMEOUT_MS = 10_000;
const DEFAULT_PROJECT_READS_TIMEOUT_MS = 5_000;

export interface MediaUsageHost {
  /** Projects this tab holds in memory; their live state wins over the server's copy. */
  getLoadedProjects(): readonly Project[];
  /** The project the deletion is made from, listed as "This project"; null without one. */
  openProjectId: string | null;
}

export interface MediaUsageOptions {
  /** How long the server check may take before the report says it could not finish. */
  serverTimeoutMs?: number;
  /** How long reading closed projects may take; projects not read by then are listed as the server counts them. */
  projectReadsTimeoutMs?: number;
  signal?: AbortSignal;
}

/**
 * A signal that aborts with `signal` or once `ms` pass; `clear` ends the timer when the stage finishes. A plain timer,
 * not `AbortSignal.timeout`: a timeout signal reachable only through `AbortSignal.any` can be garbage collected before
 * it fires (Node does, under memory pressure), and a stage would then wait on it forever.
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

const isTruncated = (entry: GalleryMediaUsageDocument): boolean =>
  entry.imageCount > entry.imageNames.length || entry.videoCount > entry.videoNames.length;

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

/**
 * Reads a closed project the way the editor would load it, without opening it; null when it cannot be loaded or is
 * past the read bound.
 */
const readClosedProject = async (projectId: string, signal: AbortSignal): Promise<Project | null> => {
  const record = await getProjectWithinSize(projectId, MEDIA_USAGE_MAX_PROJECT_READ_BYTES, signal);
  const loaded = record ? deserializeProjectRecord(record) : null;

  return loaded?.status === 'loaded' ? loaded.project : null;
};

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

/** The checked refs a project's surfaces use, by kind and name. */
const usedKeysOf = (usage: readonly MediaUsage[]): Set<string> =>
  new Set(usage.flatMap((entry) => entry.refs.map(toGalleryItemKey)));

const toSurfaceRows = (usage: readonly MediaUsage[]): GalleryMediaUsageSurfaceRow[] =>
  usage.map(({ label, refs, surface }) =>
    label === undefined ? { count: refs.length, surface } : { count: refs.length, label, surface }
  );

type ProjectOutcome =
  | { kind: 'scanned'; usage: MediaUsage[] }
  | { kind: 'unchecked'; entry: GalleryMediaUsageDocument };

/**
 * Where the requested media is used. For items, the server check failing (offline, an older server without the check,
 * a timeout) does not fail the report: the projects this tab holds are still scanned and the report says the rest went
 * unchecked. A board's members are known only to the server, so for a board that failure rejects. Otherwise rejects
 * only when `signal` aborts or the account changes.
 */
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
  const overServerLimit =
    requestedRefs.filter((ref) => ref.kind === 'image').length > MEDIA_USAGE_MAX_NAMES_PER_KIND ||
    requestedRefs.filter((ref) => ref.kind === 'video').length > MEDIA_USAGE_MAX_NAMES_PER_KIND;
  let server: GalleryMediaUsage | null = null;

  if (!overServerLimit) {
    const deadline = withDeadline(signal, serverTimeoutMs);

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
  }

  signal.throwIfAborted();
  assertAccountScopeCurrent(owner);

  if (server === null && request.kind === 'board') {
    // Only the server knows a board's members, so nothing here can be scanned for them.
    throw new Error('The media usage check did not answer for the board.');
  }

  const documents = server?.documents ?? [];
  const serverProjects = new Map(
    documents.filter((entry) => entry.kind === 'project').map((entry) => [entry.id, entry] as const)
  );
  const outcomes = new Map<string, ProjectOutcome>();
  const loadedProjects = host.getLoadedProjects();

  // Projects in memory: the items' own refs, or for a board the names the server found in that project (a board's
  // members are known only to the server).
  for (const project of loadedProjects) {
    const entry = serverProjects.get(project.id);

    if (request.kind === 'board' && entry && isTruncated(entry)) {
      outcomes.set(project.id, { entry, kind: 'unchecked' });
      continue;
    }

    const refs = request.kind === 'items' ? requestedRefs : entry ? refsOf(entry) : [];
    outcomes.set(project.id, { kind: 'scanned', usage: collectMediaUsage(project, refs) });
  }

  const candidates = [...serverProjects.values()].filter((entry) => !outcomes.has(entry.id));
  const readable = candidates.filter((entry) => !isTruncated(entry)).slice(0, MEDIA_USAGE_MAX_PROJECT_READS);

  for (const entry of candidates) {
    outcomes.set(entry.id, { entry, kind: 'unchecked' });
  }

  const readsDeadline = withDeadline(signal, projectReadsTimeoutMs);
  const readsSignal = readsDeadline.signal;

  await runLimited(readable, PROJECT_READ_CONCURRENCY, async (entry) => {
    try {
      readsSignal.throwIfAborted();
      const project = await readClosedProject(entry.id, readsSignal);

      if (project) {
        outcomes.set(entry.id, { kind: 'scanned', usage: collectMediaUsage(project, refsOf(entry)) });
      }
    } catch {
      // Unreadable, refused, or past the deadline: listed as the server counts it.
    }
  });
  readsDeadline.clear();

  signal.throwIfAborted();
  assertAccountScopeCurrent(owner);

  return buildReport({ documents, host, loadedProjects, outcomes, overServerLimit, request, requestedRefs, server });
};

const buildReport = ({
  documents,
  host,
  loadedProjects,
  outcomes,
  overServerLimit,
  request,
  requestedRefs,
  server,
}: {
  documents: readonly GalleryMediaUsageDocument[];
  host: MediaUsageHost;
  loadedProjects: ReturnType<MediaUsageHost['getLoadedProjects']>;
  outcomes: ReadonlyMap<string, ProjectOutcome>;
  overServerLimit: boolean;
  request: GalleryMediaUsageRequest;
  requestedRefs: readonly GalleryItemRef[];
  server: GalleryMediaUsage | null;
}): GalleryMediaUsageReport => {
  const groups: GalleryMediaUsageGroup[] = [];
  const inUse = new Set<string>();
  // Sources whose names are not all known (capped lists, other accounts) bound the count from below.
  let lowerBound = 0;
  let exact = true;
  // A server project entry replaced by a scan that dropped one of its names: the server's own in-use total, which
  // counted that name, no longer holds.
  let narrowed = false;
  const addNames = (entry: { imageCount: number; imageNames: string[]; videoCount: number; videoNames: string[] }) => {
    entry.imageNames.forEach((name) => inUse.add(toGalleryItemKey({ kind: 'image', name })));
    entry.videoNames.forEach((name) => inUse.add(toGalleryItemKey({ kind: 'video', name })));
    if (entry.imageCount > entry.imageNames.length || entry.videoCount > entry.videoNames.length) {
      exact = false;
      lowerBound = Math.max(lowerBound, entry.imageCount + entry.videoCount);
    }
  };
  const dropsServerNames = (projectId: string, used: ReadonlySet<string>): boolean => {
    const entry = documents.find((document) => document.kind === 'project' && document.id === projectId);

    // A capped list hides names the scan may have dropped; more counted than the scan kept means some were.
    return Boolean(
      entry &&
      (refsOf(entry).some((ref) => !used.has(toGalleryItemKey(ref))) || entry.imageCount + entry.videoCount > used.size)
    );
  };
  const projectGroup = (projectId: string, name: string | null): GalleryMediaUsageGroup | null => {
    const outcome = outcomes.get(projectId);

    if (!outcome) {
      return null;
    }
    if (outcome.kind === 'unchecked') {
      addNames(outcome.entry);
      return { count: outcome.entry.imageCount + outcome.entry.videoCount, kind: 'projectUnchecked', name };
    }

    const used = usedKeysOf(outcome.usage);

    if (dropsServerNames(projectId, used)) {
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
        groups.push(toGroup(entry));
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

  // Items: what was asked about, as the dialog's title counts them. The server's checked counts leave out items the
  // caller may not delete, but not which ones, so this tab's scans could not be held to the same set. A board: the
  // server's count, the items deleting it removes.
  const checkedCount =
    request.kind === 'items' || !server ? requestedRefs.length : server.checkedImageCount + server.checkedVideoCount;
  const serverInUse = server && !narrowed ? server.inUseImageCount + server.inUseVideoCount : 0;
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
    serverCheck: server ? 'complete' : overServerLimit ? 'tooManyItems' : 'unavailable',
  };
};

/**
 * Whether a project this tab holds uses any of `refs` now: a last look before deleting without asking, since a use
 * added while the check ran is not in its answer.
 */
export const isUsedInLoadedProjects = (refs: readonly GalleryItemRef[], host: MediaUsageHost): boolean =>
  host.getLoadedProjects().some((project) => collectMediaUsage(project, refs).length > 0);
