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
 * Track project selections, recent generations, and locked upscale inputs cleared by removal; query-cache rollback
 * cannot restore widget state.
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
  { key: 'references', widgetId: 'video' },
  { key: 'sourceVideo', widgetId: 'video' },
];

/**
 * Skip restoration when a mutually exclusive rival slot was filled after optimistic removal; bypassing setters
 * must not recreate an invalid pair.
 */
const EXCLUSIVE_RIVAL_FIELDS: ReadonlyArray<{ key: string; rivalKey: string; widgetId: WidgetTypeId }> = [
  { key: 'firstFrameImage', rivalKey: 'sourceVideo', widgetId: 'video' },
  { key: 'firstFrameImage', rivalKey: 'references', widgetId: 'video' },
  { key: 'lastFrameImage', rivalKey: 'references', widgetId: 'video' },
  { key: 'sourceVideo', rivalKey: 'firstFrameImage', widgetId: 'video' },
  { key: 'references', rivalKey: 'firstFrameImage', widgetId: 'video' },
  { key: 'references', rivalKey: 'lastFrameImage', widgetId: 'video' },
  // sourceVideo and references coexist for Ref2VA reference-extend; treating them as rivals would lose valid data
  // on rollback.
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
 * Immediately after optimistic mutation, capture only changed fields with before/after values so rollback can
 * detect subsequent writes.
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
 * Restore only values still equal to the optimistic mutation's output, merging patches per project/widget. Drop
 * disappeared projects before comparison so undefined cannot falsely match.
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
    const rivals = EXCLUSIVE_RIVAL_FIELDS.filter(
      (field) => field.widgetId === entry.widgetId && field.key === entry.key
    );
    // An ordered-list slot (the video references) is "occupied" only when non-empty; a
    // nullable slot when non-null.
    const isOccupied = (value: unknown): boolean =>
      Array.isArray(value) ? value.length > 0 : value !== null && value !== undefined;

    if (
      rivals.length > 0 &&
      isOccupied(entry.before) &&
      rivals.some((rival) =>
        isOccupied(getProjectWidgetValues(projectsById.get(entry.projectId)!, entry.widgetId)[rival.rivalKey])
      )
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
 * Read board/star state from project selection, comparison, or recent generations; newly generated items may not
 * exist in list caches yet.
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
