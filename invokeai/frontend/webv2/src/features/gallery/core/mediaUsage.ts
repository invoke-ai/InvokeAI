/** The server answer is conservative (it counts selections, recents and last runs); hosts narrow it first. */

import type { GalleryItemRef } from './items';

/** A board covers every item deleting it removes. */
export type GalleryMediaUsageRequest =
  | { kind: 'items'; refs: readonly GalleryItemRef[] }
  | { boardId: string; kind: 'board' };

/** The lists are capped; the counts are true. */
export interface GalleryMediaUsageNames {
  imageCount: number;
  imageNames: string[];
  videoCount: number;
  videoNames: string[];
}

export type GalleryMediaUsageDocumentKind = 'legacyEditor' | 'project' | 'quarantinedProject' | 'workflow';

/** `other` stands for a kind this build does not know. */
export interface GalleryMediaUsageDocument extends GalleryMediaUsageNames {
  id: string;
  kind: GalleryMediaUsageDocumentKind | 'other';
  name: string | null;
  /** Null for other kinds, or when not known. */
  revision: number | null;
  /** The caller's workspace revision the names were read at; null while the caller has none, and for non-projects. */
  workspaceRevision: number | null;
  /** A batched check could not list it in every request, so its names and counts are partial. */
  namesIncomplete?: true;
}

export interface GalleryMediaUsage {
  checkedImageCount: number;
  checkedVideoCount: number;
  /** At most a hundred of `documentsTotal`. */
  documents: GalleryMediaUsageDocument[];
  documentsTotal: number;
  inUseImageCount: number;
  inUseVideoCount: number;
  otherAccounts: { documentCount: number; imageCount: number; queueItemCount: number; videoCount: number };
  queue: GalleryMediaUsageNames & { itemCount: number };
}

/** In panel order; the labels are the Gallery's translations. */
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
  /** The scan fails safe on fields it has no surface for. */
  | 'otherProjectState';

export interface GalleryMediaUsageSurfaceRow {
  count: number;
  label?: string;
  surface: GalleryMediaUsageSurface;
}

/** `countAtLeast`: the server could not name all of the source's uses. */
export type GalleryMediaUsageGroup = { countAtLeast?: true } &
  /** `openProject` is the project the deletion is made from. */
  (
    | { count: number; kind: 'openProject'; savedCopyUnchecked?: true; surfaces: GalleryMediaUsageSurfaceRow[] }
    | {
        count: number;
        kind: 'project';
        name: string | null;
        /** The saved version uses more but could not be checked; `count` includes it, `surfaces` does not. */
        savedCopyUnchecked?: true;
        surfaces: GalleryMediaUsageSurfaceRow[];
      }
    | { count: number; kind: 'projectUnchecked'; name: string | null }
    | { count: number; kind: 'quarantinedProject'; name: string | null }
    | { count: number; kind: 'workflow'; name: string | null }
    | { count: number; kind: 'legacyEditor' }
    | { count: number; kind: 'otherDocument'; name: string | null }
    | { documentCount: number; kind: 'moreDocuments' }
    | { count: number; jobCount: number; kind: 'queue' }
    | { count: number; kind: 'otherAccounts' }
  );

export interface GalleryMediaUsageReport {
  /** For a board, as the server counted the items deleting it removes. */
  checkedCount: number;
  groups: GalleryMediaUsageGroup[];
  /** A lower bound where the server's name lists were capped. */
  inUseCount: number;
  onlyImages: boolean;
  scope: 'board' | 'items';
  /** `unavailable`: only projects open in this tab were checked. A board's report always has the server's answer. */
  serverCheck: 'complete' | 'unavailable';
}

export type GalleryMediaUsageSettledCheck = { status: 'failed' } | { report: GalleryMediaUsageReport; status: 'ready' };

/** A retry keeps the previous answer on screen (`previous`) while it runs. */
export type GalleryMediaUsageCheck =
  | { previous?: GalleryMediaUsageSettledCheck; status: 'checking' }
  | GalleryMediaUsageSettledCheck;

/** Nothing uses the media and every source was checked. */
export const isGalleryMediaUnused = (report: GalleryMediaUsageReport): boolean =>
  report.groups.length === 0 && report.serverCheck === 'complete';
