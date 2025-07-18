import { clamp } from 'es-toolkit';
import type { AutoSwitchMode } from 'features/controlLayers/store/canvasSettingsSlice';
import type { ProgressImage } from 'features/nodes/types/common';
import type { MapStore } from 'nanostores';
import { atom, computed, map } from 'nanostores';
import type { ImageDTO, S } from 'services/api/types';
import { objectEntries } from 'tsafe';

import { getOutputImageName } from './shared';

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
  loadImage: (imageName: string) => Promise<HTMLImageElement>;
  onItemsChanged: (handler: (data: S['SessionQueueItem'][]) => Promise<void> | void) => () => void;
  onQueueItemStatusChanged: (handler: (data: S['QueueItemStatusChangedEvent']) => Promise<void> | void) => () => void;
  onInvocationProgress: (handler: (data: S['InvocationProgressEvent']) => Promise<void> | void) => () => void;
};

/** Progress data for a single queue item */
export type ProgressData = {
  itemId: number;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
  imageDTO: ImageDTO | null;
};

/** Combined data for the currently selected item */
export type SelectedItemData = {
  item: S['SessionQueueItem'];
  index: number;
  progressData: ProgressData;
};

/** Creates initial progress data for a queue item */
export const getInitialProgressData = (itemId: number): ProgressData => ({
  itemId,
  progressEvent: null,
  progressImage: null,
  imageDTO: null,
});
type ProgressDataMap = Record<number, ProgressData | undefined>;

/**
 * API for managing the Canvas Staging Area - a view of the image generation queue.
 * Provides reactive state management for pending, in-progress, and completed images.
 * Users can accept images to place on canvas, discard them, navigate between items,
 * and configure auto-switching behavior.
 */
export class StagingAreaApi {
  sessionId: string;
  _app: StagingAreaAppApi;
  _subscriptions = new Set<() => void>();

  constructor(sessionId: string, app: StagingAreaAppApi) {
    this.sessionId = sessionId;
    this._app = app;

    this._subscriptions.add(this._app.onItemsChanged(this.onItemsChangedEvent));
    this._subscriptions.add(this._app.onQueueItemStatusChanged(this.onQueueItemStatusChangedEvent));
    this._subscriptions.add(this._app.onInvocationProgress(this.onInvocationProgressEvent));
  }

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

  /** Total number of items in the queue. */
  $itemCount = computed([this.$items], (items) => items.length);

  /** Whether there are any items in the queue. */
  $hasItems = computed([this.$items], (items) => items.length > 0);

  /** Whether there are any pending or in-progress items. */
  $isPending = computed([this.$items], (items) =>
    items.some((item) => item.status === 'pending' || item.status === 'in_progress')
  );

  /** The currently selected queue item with its index and progress data, or null if none selected. */
  $selectedItem = computed(
    [this.$items, this.$selectedItemId, this.$progressData],
    (items, selectedItemId, progressData) => {
      if (items.length === 0) {
        return null;
      }
      if (selectedItemId === null) {
        return null;
      }
      const item = items.find(({ item_id }) => item_id === selectedItemId);
      if (!item) {
        return null;
      }

      return {
        item,
        index: items.findIndex(({ item_id }) => item_id === selectedItemId),
        progressData: progressData[selectedItemId] || getInitialProgressData(selectedItemId),
      };
    }
  );

  /** The ImageDTO of the currently selected item, or null if none available. */
  $selectedItemImageDTO = computed([this.$selectedItem], (selectedItem) => {
    return selectedItem?.progressData.imageDTO ?? null;
  });

  /** The index of the currently selected item, or null if none selected. */
  $selectedItemIndex = computed([this.$selectedItem], (selectedItem) => {
    return selectedItem?.index ?? null;
  });

  /** Selects a queue item by ID. */
  select = (itemId: number) => {
    this.$selectedItemId.set(itemId);
    this._app.onSelect?.(itemId);
  };

  /** Selects the next item in the queue, wrapping to the first item if at the end. */
  selectNext = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const items = this.$items.get();
    const nextIndex = (selectedItem.index + 1) % items.length;
    const nextItem = items[nextIndex];
    if (!nextItem) {
      return;
    }
    this.$selectedItemId.set(nextItem.item_id);
    this._app.onSelectNext?.();
  };

  /** Selects the previous item in the queue, wrapping to the last item if at the beginning. */
  selectPrev = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const items = this.$items.get();
    const prevIndex = (selectedItem.index - 1 + items.length) % items.length;
    const prevItem = items[prevIndex];
    if (!prevItem) {
      return;
    }
    this.$selectedItemId.set(prevItem.item_id);
    this._app.onSelectPrev?.();
  };

  /** Selects the first item in the queue. */
  selectFirst = () => {
    const items = this.$items.get();
    const first = items.at(0);
    if (!first) {
      return;
    }
    this.$selectedItemId.set(first.item_id);
    this._app.onSelectFirst?.();
  };

  /** Selects the last item in the queue. */
  selectLast = () => {
    const items = this.$items.get();
    const last = items.at(-1);
    if (!last) {
      return;
    }
    this.$selectedItemId.set(last.item_id);
    this._app.onSelectLast?.();
  };

  /** Discards the currently selected item and selects the next available item. */
  discardSelected = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const items = this.$items.get();
    const nextIndex = clamp(selectedItem.index + 1, 0, items.length - 1);
    const nextItem = items[nextIndex];
    if (nextItem) {
      this.$selectedItemId.set(nextItem.item_id);
    } else {
      this.$selectedItemId.set(null);
    }
    this._app.onDiscard?.(selectedItem.item);
  };

  /** Whether the discard selected action is enabled. */
  $discardSelectedIsEnabled = computed([this.$selectedItem], (selectedItem) => {
    if (selectedItem === null) {
      return false;
    }
    return true;
  });

  /** Discards all items in the queue. */
  discardAll = () => {
    this.$selectedItemId.set(null);
    this._app.onDiscardAll?.();
  };

  /** Accepts the currently selected item if an image is available. */
  acceptSelected = () => {
    const selectedItem = this.$selectedItem.get();
    if (selectedItem === null) {
      return;
    }
    const progressData = this.$progressData.get();
    const datum = progressData[selectedItem.item.item_id];
    if (!datum || !datum.imageDTO) {
      return;
    }
    this._app.onAccept?.(selectedItem.item, datum.imageDTO);
  };

  /** Whether the accept selected action is enabled. */
  $acceptSelectedIsEnabled = computed([this.$selectedItem, this.$progressData], (selectedItem, progressData) => {
    if (selectedItem === null) {
      return false;
    }
    const datum = progressData[selectedItem.item.item_id];
    return !!datum && !!datum.imageDTO;
  });

  /** Sets the auto-switch mode. */
  setAutoSwitch = (mode: AutoSwitchMode) => {
    this._app.onAutoSwitchChange?.(mode);
  };

  /** Handles invocation progress events from the WebSocket. */
  onInvocationProgressEvent = (data: S['InvocationProgressEvent']) => {
    if (data.destination !== this.sessionId) {
      return;
    }
    setProgress(this.$progressData, data);
  };

  /** Handles queue item status change events from the WebSocket. */
  onQueueItemStatusChangedEvent = (data: S['QueueItemStatusChangedEvent']) => {
    if (data.destination !== this.sessionId) {
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
       * Then in the progress data sync effect, we process the queue item load its image.
       */
      this.$lastCompletedItemId.set(data.item_id);
    }
    if (data.status === 'in_progress' && this._app.getAutoSwitch() === 'switch_on_start') {
      this.$lastStartedItemId.set(data.item_id);
    }
  };

  /**
   * Handles queue items changed events. Updates items, manages progress data,
   * handles auto-selection, and implements auto-switch behavior.
   */
  onItemsChangedEvent = async (items: S['SessionQueueItem'][]) => {
    const oldItems = this.$items.get();

    if (items === oldItems) {
      return;
    }

    if (items.length === 0) {
      // If there are no items, cannot have a selected item.
      this.$selectedItemId.set(null);
    } else if (this.$selectedItemId.get() === null && items.length > 0) {
      // If there is no selected item but there are items, select the first one.
      this.$selectedItemId.set(items[0]?.item_id ?? null);
    }

    const progressData = this.$progressData.get();

    const toDelete: number[] = [];
    const toUpdate: ProgressData[] = [];

    for (const [id, datum] of objectEntries(progressData)) {
      if (!datum) {
        toDelete.push(id);
        continue;
      }
      const item = items.find(({ item_id }) => item_id === datum.itemId);
      if (!item) {
        toDelete.push(datum.itemId);
      } else if (item.status === 'canceled' || item.status === 'failed') {
        toUpdate.push({
          ...datum,
          progressEvent: null,
          progressImage: null,
          imageDTO: null,
        });
      }
    }

    for (const item of items) {
      const datum = progressData[item.item_id];

      if (this.$lastStartedItemId.get() === item.item_id && this._app.getAutoSwitch() === 'switch_on_start') {
        this.$selectedItemId.set(item.item_id);
        this.$lastStartedItemId.set(null);
      }

      if (datum?.imageDTO) {
        continue;
      }
      const outputImageName = getOutputImageName(item);
      if (!outputImageName) {
        continue;
      }
      const imageDTO = await this._app.getImageDTO(outputImageName);
      if (!imageDTO) {
        continue;
      }

      // This is the load logic mentioned in the comment in the QueueItemStatusChangedEvent handler above.
      if (this.$lastCompletedItemId.get() === item.item_id && this._app.getAutoSwitch() === 'switch_on_finish') {
        this._app.loadImage(imageDTO.image_url).then(() => {
          this.$selectedItemId.set(item.item_id);
          this.$lastCompletedItemId.set(null);
        });
      }

      toUpdate.push({
        ...getInitialProgressData(item.item_id),
        ...datum,
        imageDTO,
      });
    }

    for (const itemId of toDelete) {
      this.$progressData.setKey(itemId, undefined);
    }

    for (const datum of toUpdate) {
      this.$progressData.setKey(datum.itemId, datum);
    }

    this.$items.set(items);
  };

  /** Creates a computed value that returns true if the given item ID is selected. */
  buildIsSelectedComputed = (itemId: number) => {
    return computed([this.$selectedItemId], (selectedItemId) => {
      return selectedItemId === itemId;
    });
  };

  /** Cleans up all state and unsubscribes from all events. */
  cleanup = () => {
    this.$lastStartedItemId.set(null);
    this.$lastCompletedItemId.set(null);
    this.$items.set([]);
    this.$progressData.set({});
    this.$selectedItemId.set(null);
    this._subscriptions.forEach((unsubscribe) => unsubscribe());
    this._subscriptions.clear();
  };
}

/** Updates progress data for a queue item with the latest progress event. */
const setProgress = ($progressData: MapStore<ProgressDataMap>, data: S['InvocationProgressEvent']) => {
  const progressData = $progressData.get();
  const current = progressData[data.item_id];
  if (current) {
    const next = { ...current };
    next.progressEvent = data;
    if (data.image) {
      next.progressImage = data.image;
    }
    $progressData.set({
      ...progressData,
      [data.item_id]: next,
    });
  } else {
    $progressData.set({
      ...progressData,
      [data.item_id]: {
        itemId: data.item_id,
        progressEvent: data,
        progressImage: data.image ?? null,
        imageDTO: null,
      },
    });
  }
};
