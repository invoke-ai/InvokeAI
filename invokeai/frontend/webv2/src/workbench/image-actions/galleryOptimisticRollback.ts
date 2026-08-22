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
import { isSlotUnclaimed, selectUnclaimedEntries } from '@platform/state/compareAndSwapRollback';
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
  { key: 'firstFrameImage', widgetId: 'video' },
  { key: 'lastFrameImage', widgetId: 'video' },
  { key: 'sourceVideo', widgetId: 'video' },
];

/**
 * Widget slots that are mutually exclusive by design (the video widget's
 * first-frame XOR initial-video rule, enforced by its setters). A rollback
 * restore is the one write path that bypasses those setters: after our
 * optimistic clear, the user may have claimed the RIVAL slot, and restoring
 * this one would recreate a pair no setter can produce — normalization then
 * masks one of the two, which reads as a silently wrong generation mode.
 * When the rival slot is occupied, the restore loses.
 */
const EXCLUSIVE_RIVAL_FIELDS: ReadonlyArray<{ key: string; rivalKey: string; widgetId: WidgetTypeId }> = [
  { key: 'firstFrameImage', rivalKey: 'sourceVideo', widgetId: 'video' },
  { key: 'sourceVideo', rivalKey: 'firstFrameImage', widgetId: 'video' },
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
 * Selects which snapshot entries are still safe to restore, under the shared
 * compare-and-swap rule (`@platform/state/compareAndSwapRollback`): a key
 * restores only if its current value is still what our own mutation left it
 * at. A key something else wrote to since (a generation completing, a user
 * re-selecting) is left alone rather than clobbered. Entries for the same
 * project+widget are merged into a single patch.
 *
 * A project that has since disappeared has no slot to restore, so its entries
 * drop out before the compare rather than resolving to a bare `undefined` that
 * the check could mistake for a match.
 */
export const selectRestorableGalleryWidgetPatches = (
  entries: readonly GalleryWidgetKeySnapshotEntry[],
  projects: readonly Project[]
): GalleryWidgetRestorePatch[] => {
  const projectsById = new Map(projects.map((project) => [project.id, project]));
  const patchesByGroup = new Map<string, GalleryWidgetRestorePatch>();
  const liveEntries = entries.filter((entry) => projectsById.has(entry.projectId));

  for (const entry of selectUnclaimedEntries(
    liveEntries,
    (candidate) => getProjectWidgetValues(projectsById.get(candidate.projectId)!, candidate.widgetId)[candidate.key]
  )) {
    const rival = EXCLUSIVE_RIVAL_FIELDS.find((field) => field.widgetId === entry.widgetId && field.key === entry.key);
    const isOccupied = (value: unknown): boolean => value !== null && value !== undefined;

    if (
      rival &&
      isOccupied(entry.before) &&
      isOccupied(getProjectWidgetValues(projectsById.get(entry.projectId)!, entry.widgetId)[rival.rivalKey])
    ) {
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
 * The per-item form of the shared compare-and-swap rule: a key rolls back only
 * if its current value is still what *this* mutation painted it to. A key a
 * second, later mutation already moved on from (a subsequent drag to a
 * different board, another star toggle) is excluded.
 *
 * Unknown counts as unclaimed here, unlike the widget-value case: `readCurrent`
 * looks the item up in local state that may simply not hold it, so `undefined`
 * means "nothing here to protect" rather than "someone cleared it".
 */
export const selectItemKeysUnchangedSince = <Value>(
  keys: readonly GalleryItemKey[],
  paintedValue: Value,
  readCurrent: (key: GalleryItemKey) => Value | undefined
): GalleryItemKey[] =>
  keys.filter((key) => isSlotUnclaimed(readCurrent(key), paintedValue, { treatUnknownAsUnclaimed: true }));
