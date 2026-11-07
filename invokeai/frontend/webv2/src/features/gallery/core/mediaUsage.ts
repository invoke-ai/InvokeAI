/**
 * What still uses gallery media the user is about to delete: the server's reference check as the client reads it, and
 * the report a deletion dialog shows. The server answer is conservative (it counts gallery selections, recents and
 * workflow last runs inside projects); a host that can open projects narrows it before building the report, dropping
 * only what it knows shows or records media rather than working on it.
 */

import type { GalleryItemRef } from './items';

/** What to check: these items, or every item on a board (what deleting the board with its media removes). */
export type GalleryMediaUsageRequest =
  | { kind: 'items'; refs: readonly GalleryItemRef[] }
  | { boardId: string; kind: 'board' };

/** Names a source uses among the checked media. The lists are capped; the counts are true. */
export interface GalleryMediaUsageNames {
  imageCount: number;
  imageNames: string[];
  videoCount: number;
  videoNames: string[];
}

export type GalleryMediaUsageDocumentKind = 'legacyEditor' | 'project' | 'quarantinedProject' | 'workflow';

/** One of the caller's own saved documents; `other` stands for a kind this build does not know. */
export interface GalleryMediaUsageDocument extends GalleryMediaUsageNames {
  id: string;
  kind: GalleryMediaUsageDocumentKind | 'other';
  name: string | null;
  /** For a project: the content revision its names were read at; null for other kinds, or when not known. */
  revision: number | null;
  /**
   * For a project whose `revision` is known: the caller's workspace revision its names were read at, null while the
   * caller has none there. Null for other kinds.
   */
  workspaceRevision: number | null;
}

export interface GalleryMediaUsage {
  checkedImageCount: number;
  checkedVideoCount: number;
  /** The caller's documents that use checked media, at most a hundred of `documentsTotal`. */
  documents: GalleryMediaUsageDocument[];
  documentsTotal: number;
  inUseImageCount: number;
  inUseVideoCount: number;
  /** Other accounts' documents and queue work, counted without saying whose. */
  otherAccounts: { documentCount: number; imageCount: number; queueItemCount: number; videoCount: number };
  /** The caller's waiting or running queue work that names checked media. */
  queue: GalleryMediaUsageNames & { itemCount: number };
}

/** Project surfaces that can hold media, in panel order; the labels are the Gallery's translations. */
export type GalleryMediaUsageSurface =
  | 'generateReferenceImages'
  | 'canvasRaster'
  | 'canvasControl'
  | 'canvasRegionalGuidance'
  | 'canvasInpaintMask'
  | 'canvasStaging'
  | 'canvasSnapshot'
  | 'upscale'
  | 'video'
  | 'workflow'
  /** Anything else in the project that names the media: the scan fails safe on fields it has no surface for. */
  | 'otherProjectState';

export interface GalleryMediaUsageSurfaceRow {
  /** How many checked items the surface uses. */
  count: number;
  /** The workflow's or snapshot's name, for the surfaces that have one. */
  label?: string;
  surface: GalleryMediaUsageSurface;
}

/** One line of the report; `count` is how many checked items the source uses. */
export type GalleryMediaUsageGroup =
  /** The project the deletion is made from; `savedCopyUnchecked` as for `project`. */
  | { count: number; kind: 'openProject'; savedCopyUnchecked?: true; surfaces: GalleryMediaUsageSurfaceRow[] }
  | {
      count: number;
      kind: 'project';
      name: string | null;
      /**
       * The project is open, but its saved version, which the server says uses more of the media, could not be
       * checked; `count` includes those uses, `surfaces` only what the open copy uses.
       */
      savedCopyUnchecked?: true;
      surfaces: GalleryMediaUsageSurfaceRow[];
    }
  /** A project the host could not open to see where it uses the media. */
  | { count: number; kind: 'projectUnchecked'; name: string | null }
  | { count: number; kind: 'quarantinedProject'; name: string | null }
  | { count: number; kind: 'workflow'; name: string | null }
  | { count: number; kind: 'legacyEditor' }
  | { count: number; kind: 'otherDocument'; name: string | null }
  /** Documents past the server's listing cap. */
  | { documentCount: number; kind: 'moreDocuments' }
  | { count: number; jobCount: number; kind: 'queue' }
  | { count: number; kind: 'otherAccounts' };

export interface GalleryMediaUsageReport {
  /** Items checked: the requested items, or the board's as the server counted them (the items deleting it removes). */
  checkedCount: number;
  groups: GalleryMediaUsageGroup[];
  /** Distinct checked items something uses; a lower bound where the server's name lists were capped. */
  inUseCount: number;
  /** Every checked item is an image, so the summary can say "images". */
  onlyImages: boolean;
  scope: 'board' | 'items';
  /**
   * Whether the server check ran: `unavailable` when it did not finish (offline, an older server, a timeout),
   * `tooManyItems` when the selection is past what one check takes. Either way only projects open in this tab were
   * checked, not other projects, workflows, the legacy editor, queued jobs or other accounts. A board's report always
   * has the server's answer, which alone knows the board's items.
   */
  serverCheck: 'complete' | 'tooManyItems' | 'unavailable';
}

/** What a deletion dialog shows while the check runs, when it failed outright, and once it has a report. */
/** A check that has answered: its report, or that it failed outright. */
export type GalleryMediaUsageSettledCheck = { status: 'failed' } | { report: GalleryMediaUsageReport; status: 'ready' };

/**
 * What a deletion dialog shows: the check running, failed outright, or its report. A retry runs while the previous
 * answer stays on screen (`previous`), so what is already known does not vanish for the length of the check.
 */
export type GalleryMediaUsageCheck =
  | { previous?: GalleryMediaUsageSettledCheck; status: 'checking' }
  | GalleryMediaUsageSettledCheck;

/** Whether the report gives no reason to stop and ask: nothing uses the media and every source was checked. */
export const isGalleryMediaUnused = (report: GalleryMediaUsageReport): boolean =>
  report.groups.length === 0 && report.serverCheck === 'complete';
