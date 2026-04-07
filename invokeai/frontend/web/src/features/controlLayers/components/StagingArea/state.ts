import { clamp } from 'es-toolkit';
import type { AutoSwitchMode } from 'features/controlLayers/store/canvasSettingsSlice';
import type { ProgressImage } from 'features/nodes/types/common';
import type { MapStore } from 'nanostores';
import { atom, computed, map } from 'nanostores';
import type { ImageDTO, S } from 'services/api/types';
import { objectEntries } from 'tsafe';

import { getOutputImageNames } from './shared';

/**
 * Interface for the app-level API that the StagingAreaApi depends on.
 * This provides the connection between the staging area and the rest of the application.
 */
export type StagingAreaAppApi = {
  onDiscard?: (item: S['SessionQueueItem']) => void;
  onDiscardAll?: () => void;
  onAccept?: (item: S['SessionQueueItem'], imageDTO: ImageDTO) => void;
  onSelect?: (itemId: number) => void;
  onSelectPrev?: () => void;
  onSelectNext?: () => void;
  onSelectFirst?: () => void;
  onSelectLast?: () => void;
  getAutoSwitch: () => AutoSwitchMode;
  onAutoSwitchChange?: (mode: AutoSwitchMode) => void;
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>;
  onItemsChanged: (handler: (data: S['SessionQueueItem'][]) => Promise<void> | void) => () => void;
  onQueueItemStatusChanged: (handler: (data: S['QueueItemStatusChangedEvent']) => Promise<void> | void) => () => void;
  onInvocationProgress: (handler: (data: S['InvocationProgressEvent']) => Promise<void> | void) => () => void;
};

/** Progress data for a single queue item */
export type ProgressData = {
  itemId: number;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
  imageDTOs: ImageDTO[];
  imageLoaded: boolean;
};

/** A single entry in the staging area. Each canvas_output image is a separate entry. */
export type StagingEntry = {
  item: S['SessionQueueItem'];
  imageDTO: ImageDTO | null;
  imageIndex: number;
  progressData: ProgressData;
};

/** Combined data for the currently selected entry */
export type SelectedItemData = {
  item: S['SessionQueueItem'];
  index: number;
  imageDTO: ImageDTO | null;
  progressData: ProgressData;
};

/** Creates initial progress data for a queue item */
export const getInitialProgressData = (itemId: number): ProgressData => ({
  itemId,
  progressEvent: null,
  progressImage: null,
  imageDTOs: [],
  imageLoaded: false,
});
type ProgressDataMap = Record<number, ProgressData | undefined>;

/**
 * API for managing the Canvas Staging Area - a view of the image generation queue.
 * Provides reactive state management for pending, in-progress, and completed images.
 * Each canvas_output node produces a separate entry that can be individually navigated and accepted.
 */
export class StagingAreaApi {
  /** The current session ID. */
  _sessionId: string | null = null;

  /** The app API */
  _app: StagingAreaAppApi | null = null;

  /** A set of subscriptions to be cleaned up when we are finished with a session */
  _subscriptions = new Set<() => void>();

  /** Generation counter to prevent stale async writes in onItemsChangedEvent */
  _itemsEventGeneration = 0;

  /** Item ID of the last started item. Used for auto-switch on start. */
  $lastStartedItemId = atom<number | null>(null);

  /** Item ID of the last completed item. Used for auto-switch on completion. */
  $lastCompletedItemId = atom<number | null>(null);

  /** All queue items for the current session. */
  $items = atom<S['SessionQueueItem'][]>([]);

  /** Progress data for all items including events, images, and ImageDTOs. */
  $progressData = map<ProgressDataMap>({});

  /** ID of the currently selected queue item, or null if none selected. */
  $selectedItemId = atom<number | null>(null);

  /** Index of the selected image within the selected queue item (for multi-output items). */
  $selectedImageIndex = atom<number>(0);

  /**
   * Flat list of staging entries. Each canvas_output image from a queue item becomes
   * a separate entry. Items with 0 or 1 output images produce a single entry.
   */
  $entries = computed([this.$items, this.$progressData], (items, progressData) => {
    const entries: StagingEntry[] = [];
    for (const item of items) {
      const datum = progressData[item.item_id] ?? getInitialProgressData(item.item_id);
      if (datum.imageDTOs.length <= 1) {
        entries.push({
          item,
          imageDTO: datum.imageDTOs[0] ?? null,
          imageIndex: 0,
          progressData: datum,
        });
      } else {
        for (let i = 0; i < datum.imageDTOs.length; i++) {
          const imageDTO = datum.imageDTOs[i];
          if (imageDTO) {
            entries.push({ item, imageDTO, imageIndex: i, progressData: datum });
          }
        }
      }
    }
    return entries;
  });

  /** Total number of entries (each canvas_output image counts separately). */
  $itemCount = computed([this.$entries], (entries) => entries.length);

  /** Whether there are any items in the queue. */
  $hasItems = computed([this.$items], (items) => items.length > 0);

  /** Whether there are any pending or in-progress items. */
  $isPending = computed([this.$items], (items) =>
    items.some((item) => item.status === 'pending' || item.status === 'in_progress')
  );

  /** The currently selected entry with its global index, or null if none selected. */
  $selectedItem = computed(
    [this.$entries, this.$selectedItemId, this.$selectedImageIndex],
    (entries, selectedItemId, selectedImageIndex) => {
      if (entries.length === 0 || selectedItemId === null) {
        return null;
      }

      // Find the entry matching (selectedItemId, selectedImageIndex)
      let targetEntry: StagingEntry | null = null;
      let globalIndex = -1;
      let imageIdxWithinItem = 0;

      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i]!;
        if (entry.item.item_id === selectedItemId) {
          if (imageIdxWithinItem === selectedImageIndex) {
            targetEntry = entry;
            globalIndex = i;
            break;
          }
          imageIdxWithinItem++;
        }
      }

      // Fallback: select first entry for this item
      if (!targetEntry) {
        for (let i = 0; i < entries.length; i++) {
          const entry = entries[i]!;
          if (entry.item.item_id === selectedItemId) {
            targetEntry = entry;
            globalIndex = i;
            break;
          }
        }
      }

      if (!targetEntry || globalIndex === -1) {
        return null;
      }

      return {
        item: targetEntry.item,
        index: globalIndex,
        imageDTO: targetEntry.imageDTO,
        progressData: targetEntry.progressData,
      };
    }
  );

  /** The ImageDTO of the currently selected entry, or null if none available. */
  $selectedItemImageDTO = computed([this.$selectedItem], (selectedItem) => {
    return selectedItem?.imageDTO ?? null;
  });

  /** The global entry index of the currently selected entry, or null if none selected. */
  $selectedItemIndex = computed([this.$selectedItem], (selectedItem) => {
    return selectedItem?.index ?? null;
  });

  /** Selects a queue item by ID, optionally at a specific image index. */
  select = (itemId: number, imageIndex: number = 0) => {
    this.$selectedItemId.set(itemId);
    this.$selectedImageIndex.set(imageIndex);
    this._app?.onSelect?.(itemId);
  };

  /** Selects the next entry, cycling through all entries across all items. */
  selectNext = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const entries = this.$entries.get();
    if (entries.length <= 1) {
      return;
    }
    const nextIndex = (selectedItem.index + 1) % entries.length;
    const nextEntry = entries[nextIndex];
    if (!nextEntry) {
      return;
    }
    this.$selectedItemId.set(nextEntry.item.item_id);
    this.$selectedImageIndex.set(nextEntry.imageIndex);
    this._app?.onSelectNext?.();
  };

  /** Selects the previous entry, cycling through all entries across all items. */
  selectPrev = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const entries = this.$entries.get();
    if (entries.length <= 1) {
      return;
    }
    const prevIndex = (selectedItem.index - 1 + entries.length) % entries.length;
    const prevEntry = entries[prevIndex];
    if (!prevEntry) {
      return;
    }
    this.$selectedItemId.set(prevEntry.item.item_id);
    this.$selectedImageIndex.set(prevEntry.imageIndex);
    this._app?.onSelectPrev?.();
  };

  /** Selects the first entry. */
  selectFirst = () => {
    const entries = this.$entries.get();
    const first = entries[0];
    if (!first) {
      return;
    }
    this.$selectedItemId.set(first.item.item_id);
    this.$selectedImageIndex.set(first.imageIndex);
    this._app?.onSelectFirst?.();
  };

  /** Selects the last entry. */
  selectLast = () => {
    const entries = this.$entries.get();
    const last = entries.at(-1);
    if (!last) {
      return;
    }
    this.$selectedItemId.set(last.item.item_id);
    this.$selectedImageIndex.set(last.imageIndex);
    this._app?.onSelectLast?.();
  };

  /** Discards the queue item of the currently selected entry and selects the next available entry. */
  discardSelected = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const items = this.$items.get();
    const itemIndex = items.findIndex((i) => i.item_id === selectedItem.item.item_id);
    const nextItemIndex = clamp(itemIndex + 1, 0, items.length - 1);
    const nextItem = items[nextItemIndex];
    if (nextItem) {
      this.$selectedItemId.set(nextItem.item_id);
    } else {
      this.$selectedItemId.set(null);
    }
    this.$selectedImageIndex.set(0);
    this._app?.onDiscard?.(selectedItem.item);
  };

  /** Whether the discard selected action is enabled. */
  $discardSelectedIsEnabled = computed([this.$selectedItem], (selectedItem) => {
    if (selectedItem === null) {
      return false;
    }
    return true;
  });

  /** Connects to the app, registering listeners and such */
  connectToApp = (sessionId: string, app: StagingAreaAppApi) => {
    if (this._sessionId !== sessionId) {
      this.cleanup();
      this._sessionId = sessionId;
    }
    this._app = app;

    this._subscriptions.add(this._app.onItemsChanged(this.onItemsChangedEvent));
    this._subscriptions.add(this._app.onQueueItemStatusChanged(this.onQueueItemStatusChangedEvent));
    this._subscriptions.add(this._app.onInvocationProgress(this.onInvocationProgressEvent));
  };

  /** Discards all items in the queue. */
  discardAll = () => {
    this.$selectedItemId.set(null);
    this.$selectedImageIndex.set(0);
    this._app?.onDiscardAll?.();
  };

  /** Accepts the currently selected entry if an image is available. */
  acceptSelected = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null || !selectedItem.imageDTO) {
      return;
    }
    this._app?.onAccept?.(selectedItem.item, selectedItem.imageDTO);
  };

  /** Whether the accept selected action is enabled. */
  $acceptSelectedIsEnabled = computed([this.$selectedItem], (selectedItem) => {
    return selectedItem !== null && selectedItem.imageDTO !== null;
  });

  /** Sets the auto-switch mode. */
  setAutoSwitch = (mode: AutoSwitchMode) => {
    this._app?.onAutoSwitchChange?.(mode);
  };

  /** Handles invocation progress events from the WebSocket. */
  onInvocationProgressEvent = (data: S['InvocationProgressEvent']) => {
    if (data.destination !== this._sessionId) {
      return;
    }
    setProgress(this.$progressData, data);
  };

  /** Handles queue item status change events from the WebSocket. */
  onQueueItemStatusChangedEvent = (data: S['QueueItemStatusChangedEvent']) => {
    if (data.destination !== this._sessionId) {
      return;
    }
    if (data.status === 'completed') {
      /**
       * There is an unpleasant bit of indirection here. When an item is completed, and auto-switch is set to
       * switch_on_finish, we want to load the image and switch to it. In this socket handler, we don't have
       * access to the full queue item, which we need to get the output image and load it. We get the full
       * queue items as part of the list query, so it's rather inefficient to fetch it again here.
       *
       * To reduce the number of extra network requests, we instead store this item as the last completed item.
       * Then when the image loads, it calls onImageLoaded and we switch to it then.
       */
      this.$lastCompletedItemId.set(data.item_id);
    }
    if (data.status === 'in_progress' && this._app?.getAutoSwitch() === 'switch_on_start') {
      this.$lastStartedItemId.set(data.item_id);
    }
  };

  /**
   * Handles queue items changed events. Updates items, manages progress data,
   * handles auto-selection, and implements auto-switch behavior.
   */
  onItemsChangedEvent = async (items: S['SessionQueueItem'][]) => {
    // Increment generation counter. If a newer call starts while we're awaiting,
    // we'll detect it and avoid overwriting with stale data.
    const generation = ++this._itemsEventGeneration;

    const oldItems = this.$items.get();

    if (items === oldItems) {
      return;
    }

    if (items.length === 0) {
      // If there are no items, cannot have a selected item.
      this.$selectedItemId.set(null);
      this.$selectedImageIndex.set(0);
    } else if (this.$selectedItemId.get() === null && items.length > 0) {
      // If there is no selected item but there are items, select the first one.
      this.$selectedItemId.set(items[0]?.item_id ?? null);
      this.$selectedImageIndex.set(0);
    }

    const progressData = this.$progressData.get();

    for (const [id, datum] of objectEntries(progressData)) {
      if (!datum || !items.find(({ item_id }) => item_id === datum.itemId)) {
        this.$progressData.setKey(id, undefined);
        continue;
      }
    }

    for (const item of items) {
      const datum = progressData[item.item_id];

      if (item.status === 'canceled' || item.status === 'failed') {
        this.$progressData.setKey(item.item_id, {
          ...(datum ?? getInitialProgressData(item.item_id)),
          progressEvent: null,
          progressImage: null,
          imageDTOs: [],
        });
        continue;
      }

      if (item.status === 'in_progress') {
        if (this.$lastStartedItemId.get() === item.item_id && this._app?.getAutoSwitch() === 'switch_on_start') {
          this.$selectedItemId.set(item.item_id);
          this.$selectedImageIndex.set(0);
          this.$lastStartedItemId.set(null);
        }
        continue;
      }

      if (item.status === 'completed') {
        const outputImageNames = getOutputImageNames(item);
        if (outputImageNames.length === 0) {
          continue;
        }
        // Check current progress data (not the snapshot) to account for concurrent updates
        const currentDatum = this.$progressData.get()[item.item_id];
        if (currentDatum && currentDatum.imageDTOs.length === outputImageNames.length) {
          continue;
        }
        const imageDTOs: ImageDTO[] = [];
        for (const imageName of outputImageNames) {
          const imageDTO = await this._app?.getImageDTO(imageName);
          if (imageDTO) {
            imageDTOs.push(imageDTO);
          }
        }
        if (imageDTOs.length === 0) {
          continue;
        }

        // After async work, check if a newer event has started processing.
        // If so, abort to let the newer call handle the update with fresher data.
        if (generation !== this._itemsEventGeneration) {
          return;
        }

        // Re-read progress data to avoid overwriting a better result from a concurrent call
        const latestDatum = this.$progressData.get()[item.item_id];
        if (latestDatum && latestDatum.imageDTOs.length >= imageDTOs.length) {
          continue;
        }

        this.$progressData.setKey(item.item_id, {
          ...(latestDatum ?? getInitialProgressData(item.item_id)),
          imageDTOs,
        });
      }
    }

    // After async work, check if a newer event has started processing
    if (generation !== this._itemsEventGeneration) {
      return;
    }

    const selectedItemId = this.$selectedItemId.get();
    if (selectedItemId !== null && !items.find(({ item_id }) => item_id === selectedItemId)) {
      // If the selected item no longer exists, select the next best item.
      // Prefer the next item in the list - must check oldItems to determine this
      const nextItemIndex = oldItems.findIndex(({ item_id }) => item_id === selectedItemId);
      if (nextItemIndex !== -1) {
        const nextItem = items[nextItemIndex] ?? items[nextItemIndex - 1];
        if (nextItem) {
          this.$selectedItemId.set(nextItem.item_id);
          this.$selectedImageIndex.set(0);
        }
      } else {
        // Next, if there is an in-progress item, select that.
        const inProgressItem = items.find(({ status }) => status === 'in_progress');
        if (inProgressItem) {
          this.$selectedItemId.set(inProgressItem.item_id);
          this.$selectedImageIndex.set(0);
        }
        // Finally just select the first item.
        this.$selectedItemId.set(items[0]?.item_id ?? null);
        this.$selectedImageIndex.set(0);
      }
    }

    this.$items.set(items);
  };

  onImageLoaded = (itemId: number) => {
    const item = this.$items.get().find(({ item_id }) => item_id === itemId);
    if (!item) {
      return;
    }
    // This is the load logic mentioned in the comment in the QueueItemStatusChangedEvent handler above.
    if (this.$lastCompletedItemId.get() === item.item_id && this._app?.getAutoSwitch() === 'switch_on_finish') {
      this.$selectedItemId.set(item.item_id);
      this.$selectedImageIndex.set(0);
      this.$lastCompletedItemId.set(null);
    }
    const datum = this.$progressData.get()[item.item_id];
    this.$progressData.setKey(item.item_id, {
      ...(datum ?? getInitialProgressData(item.item_id)),
      imageLoaded: true,
    });
  };

  /** Creates a computed value that returns true if the given item ID and image index is selected. */
  buildIsSelectedComputed = (itemId: number, imageIndex: number = 0) => {
    return computed([this.$selectedItemId, this.$selectedImageIndex], (selectedItemId, selectedImageIndex) => {
      return selectedItemId === itemId && selectedImageIndex === imageIndex;
    });
  };

  /** Cleans up all state and unsubscribes from all events. */
  cleanup = () => {
    this._itemsEventGeneration++;
    this.$lastStartedItemId.set(null);
    this.$lastCompletedItemId.set(null);
    this.$items.set([]);
    this.$progressData.set({});
    this.$selectedItemId.set(null);
    this.$selectedImageIndex.set(0);
    this._subscriptions.forEach((unsubscribe) => unsubscribe());
    this._subscriptions.clear();
  };
}

/** Updates progress data for a queue item with the latest progress event. */
const setProgress = ($progressData: MapStore<ProgressDataMap>, data: S['InvocationProgressEvent']) => {
  const progressData = $progressData.get();
  const current = progressData[data.item_id];
  const next = { ...(current ?? getInitialProgressData(data.item_id)) };
  next.progressEvent = data;
  if (data.image) {
    next.progressImage = data.image;
  }
  $progressData.set({
    ...progressData,
    [data.item_id]: next,
  });
};
