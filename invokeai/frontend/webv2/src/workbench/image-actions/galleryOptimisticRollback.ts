import type {
  GalleryImage,
  GalleryItem,
  GalleryItemKey,
  GalleryItemRef,
  GeneratedImageContract,
} from '@features/gallery';
import type { Project } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery';
import {
  getBoundedRecentImages,
  getGalleryCompareImage,
  getSelectedGalleryItemFromValues,
} from '@features/gallery/contracts';
import { getProjectWidgetValues } from '@workbench/widgetState';

/**
 * Fields `removeGalleryItemsFromAllProjects` (workbenchState.ts) can mutate,
 * across every project: it strips a removed item from any project's
 * recent-generation overlay and selection, and clears the upscale widget's
 * locked input if it pointed at a removed image. None of that is visible to
 * `patchGalleryItemCaches`, which only ever touches the query cache.
 */
const TRACKED_GALLERY_REMOVAL_FIELDS: ReadonlyArray<{ key: string; widgetId: WidgetTypeId }> = [
  { key: 'compareImage', widgetId: 'gallery' },
  { key: 'recentImages', widgetId: 'gallery' },
  { key: 'selectedImage', widgetId: 'gallery' },
  { key: 'selectedImageName', widgetId: 'gallery' },
  { key: 'selectedImageNames', widgetId: 'gallery' },
  { key: 'inputImage', widgetId: 'upscale' },
];

const trackedFieldKey = (projectId: string, widgetId: WidgetTypeId, key: string): string =>
  `${projectId}:${widgetId}:${key}`;

const widgetGroupKey = (projectId: string, widgetId: WidgetTypeId): string => `${projectId}:${widgetId}`;

export type GalleryWidgetKeyValueMap = Map<string, unknown>;

/** Reads every tracked field's current value, per project. Call before the optimistic mutation. */
export const captureGalleryWidgetKeyValues = (projects: readonly Project[]): GalleryWidgetKeyValueMap => {
  const values: GalleryWidgetKeyValueMap = new Map();

  for (const project of projects) {
    for (const field of TRACKED_GALLERY_REMOVAL_FIELDS) {
      const widgetValues = getProjectWidgetValues(project, field.widgetId);

      values.set(trackedFieldKey(project.id, field.widgetId, field.key), widgetValues[field.key]);
    }
  }

  return values;
};

export interface GalleryWidgetKeySnapshotEntry {
  after: unknown;
  before: unknown;
  key: string;
  projectId: string;
  widgetId: WidgetTypeId;
}

/**
 * Diffs the tracked fields' current value against a `before` snapshot and
 * returns only the entries the optimistic mutation actually changed, each
 * carrying both the pre-mutation and post-mutation ("after") value — the same
 * shape `patchGalleryItemCaches`'s rollback closure captures
 * (`queryCache.ts` `ItemCacheRollbackEntry`), so the restore can apply the
 * same conflict rule: only put a key back if nothing else wrote to it since.
 * Call immediately after the optimistic mutation.
 */
export const diffGalleryWidgetKeyValues = (
  before: GalleryWidgetKeyValueMap,
  projects: readonly Project[]
): GalleryWidgetKeySnapshotEntry[] => {
  const entries: GalleryWidgetKeySnapshotEntry[] = [];

  for (const project of projects) {
    for (const field of TRACKED_GALLERY_REMOVAL_FIELDS) {
      const mapKey = trackedFieldKey(project.id, field.widgetId, field.key);

      if (!before.has(mapKey)) {
        continue;
      }

      const beforeValue = before.get(mapKey);
      const afterValue = getProjectWidgetValues(project, field.widgetId)[field.key];

      if (afterValue === beforeValue) {
        continue;
      }

      entries.push({
        after: afterValue,
        before: beforeValue,
        key: field.key,
        projectId: project.id,
        widgetId: field.widgetId,
      });
    }
  }

  return entries;
};

export interface GalleryWidgetRestorePatch {
  projectId: string;
  values: Record<string, unknown>;
  widgetId: WidgetTypeId;
}

/**
 * Selects which snapshot entries are still safe to restore: a key restores
 * only if its current value is still reference-equal to what our own
 * mutation left it at — mirroring `patchGalleryItemCaches`'s rollback rule
 * (`if (client.getQueryData(queryKey) === after)`). A key something else
 * wrote to since (a generation completing, a user re-selecting) is left
 * alone rather than clobbered. Entries for the same project+widget are
 * merged into a single patch.
 */
export const selectRestorableGalleryWidgetPatches = (
  entries: readonly GalleryWidgetKeySnapshotEntry[],
  projects: readonly Project[]
): GalleryWidgetRestorePatch[] => {
  const projectsById = new Map(projects.map((project) => [project.id, project]));
  const patchesByGroup = new Map<string, GalleryWidgetRestorePatch>();

  for (const entry of entries) {
    const project = projectsById.get(entry.projectId);

    if (!project) {
      continue;
    }

    const currentValue = getProjectWidgetValues(project, entry.widgetId)[entry.key];

    if (currentValue !== entry.after) {
      continue;
    }

    const groupKey = widgetGroupKey(entry.projectId, entry.widgetId);
    const patch = patchesByGroup.get(groupKey) ?? { projectId: entry.projectId, values: {}, widgetId: entry.widgetId };

    patch.values[entry.key] = entry.before;
    patchesByGroup.set(groupKey, patch);
  }

  return [...patchesByGroup.values()];
};

export interface GalleryStoreKnownItemFields {
  boardId: string;
  starred: boolean;
}

/**
 * Reads the board and starred state of each requested item from whichever
 * project widget state holds it locally: the recent-generation overlay, the
 * current selection, or the compare slot. `getGalleryItemBoardIdsFromCaches`
 * /`getGalleryItemStarredFromCaches` only see items a list query has already
 * fetched; a just-generated image can be known *only* here.
 */
export const collectGalleryStoreKnownItemFields = (
  projects: readonly Project[],
  refs: readonly GalleryItemRef[]
): Map<GalleryItemKey, GalleryStoreKnownItemFields> => {
  const wanted = new Set(refs.map(toGalleryItemKey));
  const found = new Map<GalleryItemKey, GalleryStoreKnownItemFields>();
  const record = (item: GalleryItem) => {
    const key = toGalleryItemKey(item);

    if (wanted.has(key) && !found.has(key)) {
      found.set(key, { boardId: item.boardId, starred: item.starred });
    }
  };

  for (const project of projects) {
    if (found.size === wanted.size) {
      break;
    }

    const values = getProjectWidgetValues(project, 'gallery');

    for (const image of getBoundedRecentImages(values.recentImages)) {
      record(legacyGeneratedImageToGalleryItem(image as GeneratedImageContract & Partial<GalleryImage>));
    }

    const selectedItem = getSelectedGalleryItemFromValues(values);

    if (selectedItem) {
      record(selectedItem);
    }

    const compareImage = getGalleryCompareImage(values);

    if (compareImage) {
      record(legacyGeneratedImageToGalleryItem(compareImage));
    }
  }

  return found;
};

/**
 * Filters `keys` down to those still safe to roll back: a key restores only
 * if its current value (as read by `readCurrent`) is either unknown (nothing
 * locally to protect) or still equal to `paintedValue` — the value *this*
 * optimistic mutation painted it to. A key a second, later mutation already
 * moved on from (e.g. a subsequent drag to a different board, or another
 * star toggle) is excluded, the per-item analogue of
 * `selectRestorableGalleryWidgetPatches`'s reference-equality CAS check and
 * `patchGalleryItemCaches`'s own `if (current === after)` rollback rule.
 */
export const selectItemKeysUnchangedSince = <Value>(
  keys: readonly GalleryItemKey[],
  paintedValue: Value,
  readCurrent: (key: GalleryItemKey) => Value | undefined
): GalleryItemKey[] =>
  keys.filter((key) => {
    const current = readCurrent(key);

    return current === undefined || current === paintedValue;
  });
