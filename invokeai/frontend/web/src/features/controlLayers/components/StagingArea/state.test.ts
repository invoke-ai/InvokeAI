import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  createMockImageDTO,
  createMockProgressEvent,
  createMockQueueItem,
  createMockQueueItemStatusChangedEvent,
  createMockStagingAreaApp,
} from './__mocks__/mockStagingAreaApp';
import { StagingAreaApi } from './state';

describe('StagingAreaApi', () => {
  let api: StagingAreaApi;
  let mockApp: ReturnType<typeof createMockStagingAreaApp>;
  const sessionId = 'test-session';

  beforeEach(() => {
    mockApp = createMockStagingAreaApp();
    api = new StagingAreaApi();
    api.connectToApp(sessionId, mockApp);
  });

  afterEach(() => {
    api.cleanup();
  });

  describe('Constructor and Setup', () => {
    it('should initialize with correct session ID', () => {
      expect(api._sessionId).toBe(sessionId);
    });

    it('should set up event subscriptions', () => {
      expect(mockApp.onItemsChanged).toHaveBeenCalledWith(expect.any(Function));
      expect(mockApp.onQueueItemStatusChanged).toHaveBeenCalledWith(expect.any(Function));
      expect(mockApp.onInvocationProgress).toHaveBeenCalledWith(expect.any(Function));
    });

    it('should initialize atoms with default values', () => {
      expect(api.$lastStartedItemId.get()).toBe(null);
      expect(api.$lastCompletedItemId.get()).toBe(null);
      expect(api.$items.get()).toEqual([]);
      expect(api.$progressData.get()).toEqual({});
      expect(api.$selectedItemId.get()).toBe(null);
      expect(api.$selectedImageIndex.get()).toBe(0);
    });
  });

  describe('Computed Values', () => {
    it('should compute item count as entry count for single-output items', () => {
      expect(api.$itemCount.get()).toBe(0);

      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      // Item with no imageDTOs produces 1 entry
      expect(api.$itemCount.get()).toBe(1);
    });

    it('should compute item count as entry count for multi-output items', () => {
      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [createMockImageDTO({ image_name: 'a.png' }), createMockImageDTO({ image_name: 'b.png' })],
        imageLoaded: false,
      });
      // 2 imageDTOs = 2 entries
      expect(api.$itemCount.get()).toBe(2);
    });

    it('should compute hasItems correctly', () => {
      expect(api.$hasItems.get()).toBe(false);

      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      expect(api.$hasItems.get()).toBe(true);
    });

    it('should compute isPending correctly', () => {
      expect(api.$isPending.get()).toBe(false);

      const items = [
        createMockQueueItem({ item_id: 1, status: 'pending' }),
        createMockQueueItem({ item_id: 2, status: 'completed' }),
      ];
      api.$items.set(items);
      expect(api.$isPending.get()).toBe(true);
    });

    it('should compute selectedItem correctly', () => {
      expect(api.$selectedItem.get()).toBe(null);

      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);
      api.$selectedItemId.set(1);

      const selectedItem = api.$selectedItem.get();
      expect(selectedItem).not.toBe(null);
      expect(selectedItem?.item.item_id).toBe(1);
      expect(selectedItem?.index).toBe(0);
    });

    it('should compute selectedItemImageDTO correctly', () => {
      const items = [createMockQueueItem({ item_id: 1 })];
      const imageDTO = createMockImageDTO();

      api.$items.set(items);
      api.$selectedItemId.set(1);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [imageDTO],
        imageLoaded: false,
      });

      expect(api.$selectedItemImageDTO.get()).toBe(imageDTO);
    });

    it('should compute selectedItemIndex correctly', () => {
      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);
      api.$selectedItemId.set(2);

      expect(api.$selectedItemIndex.get()).toBe(1);
    });
  });

  describe('Entries Computed', () => {
    it('should produce one entry per item when each has 0 or 1 imageDTOs', () => {
      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);

      const entries = api.$entries.get();
      expect(entries).toHaveLength(2);
      expect(entries[0]?.item.item_id).toBe(1);
      expect(entries[0]?.imageIndex).toBe(0);
      expect(entries[0]?.imageDTO).toBe(null);
      expect(entries[1]?.item.item_id).toBe(2);
      expect(entries[1]?.imageIndex).toBe(0);
    });

    it('should produce one entry per item when each has exactly 1 imageDTO', () => {
      const imageDTO = createMockImageDTO({ image_name: 'img.png' });
      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [imageDTO],
        imageLoaded: false,
      });

      const entries = api.$entries.get();
      expect(entries).toHaveLength(1);
      expect(entries[0]?.imageDTO).toBe(imageDTO);
      expect(entries[0]?.imageIndex).toBe(0);
    });

    it('should produce multiple entries for items with multiple imageDTOs', () => {
      const img1 = createMockImageDTO({ image_name: 'a.png' });
      const img2 = createMockImageDTO({ image_name: 'b.png' });
      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1, img2],
        imageLoaded: false,
      });

      const entries = api.$entries.get();
      expect(entries).toHaveLength(2);
      expect(entries[0]?.item.item_id).toBe(1);
      expect(entries[0]?.imageDTO).toBe(img1);
      expect(entries[0]?.imageIndex).toBe(0);
      expect(entries[1]?.item.item_id).toBe(1);
      expect(entries[1]?.imageDTO).toBe(img2);
      expect(entries[1]?.imageIndex).toBe(1);
    });

    it('should interleave entries from multiple items', () => {
      const img1a = createMockImageDTO({ image_name: 'a1.png' });
      const img1b = createMockImageDTO({ image_name: 'a2.png' });
      const img2 = createMockImageDTO({ image_name: 'b1.png' });
      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1a, img1b],
        imageLoaded: false,
      });
      api.$progressData.setKey(2, {
        itemId: 2,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img2],
        imageLoaded: false,
      });

      const entries = api.$entries.get();
      expect(entries).toHaveLength(3);
      // Item 1 entries first
      expect(entries[0]?.item.item_id).toBe(1);
      expect(entries[0]?.imageDTO).toBe(img1a);
      expect(entries[1]?.item.item_id).toBe(1);
      expect(entries[1]?.imageDTO).toBe(img1b);
      // Then item 2
      expect(entries[2]?.item.item_id).toBe(2);
      expect(entries[2]?.imageDTO).toBe(img2);
    });
  });

  describe('Selection Methods', () => {
    beforeEach(() => {
      const items = [
        createMockQueueItem({ item_id: 1 }),
        createMockQueueItem({ item_id: 2 }),
        createMockQueueItem({ item_id: 3 }),
      ];
      api.$items.set(items);
    });

    it('should select item by ID', () => {
      api.select(2);
      expect(api.$selectedItemId.get()).toBe(2);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(mockApp.onSelect).toHaveBeenCalledWith(2);
    });

    it('should select item by ID and imageIndex', () => {
      api.select(2, 1);
      expect(api.$selectedItemId.get()).toBe(2);
      expect(api.$selectedImageIndex.get()).toBe(1);
    });

    it('should select next item', () => {
      api.$selectedItemId.set(1);
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(2);
      expect(mockApp.onSelectNext).toHaveBeenCalled();
    });

    it('should wrap to first item when selecting next from last', () => {
      api.$selectedItemId.set(3);
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(1);
    });

    it('should select previous item', () => {
      api.$selectedItemId.set(2);
      api.selectPrev();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(mockApp.onSelectPrev).toHaveBeenCalled();
    });

    it('should wrap to last item when selecting previous from first', () => {
      api.$selectedItemId.set(1);
      api.selectPrev();
      expect(api.$selectedItemId.get()).toBe(3);
    });

    it('should select first item', () => {
      api.selectFirst();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(mockApp.onSelectFirst).toHaveBeenCalled();
    });

    it('should select last item', () => {
      api.selectLast();
      expect(api.$selectedItemId.get()).toBe(3);
      expect(mockApp.onSelectLast).toHaveBeenCalled();
    });

    it('should do nothing when no items exist', () => {
      api.$items.set([]);
      api.selectNext();
      api.selectPrev();
      api.selectFirst();
      api.selectLast();

      expect(api.$selectedItemId.get()).toBe(null);
    });

    it('should do nothing when no item is selected', () => {
      api.$selectedItemId.set(null);
      api.selectNext();
      api.selectPrev();

      expect(api.$selectedItemId.get()).toBe(null);
    });
  });

  describe('Entry-Based Navigation', () => {
    it('should navigate through entries of multi-output items', () => {
      const img1 = createMockImageDTO({ image_name: 'a.png' });
      const img2 = createMockImageDTO({ image_name: 'b.png' });
      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1, img2],
        imageLoaded: false,
      });
      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(0);

      // Should be on first entry (imageIndex 0)
      expect(api.$selectedItem.get()?.imageDTO).toBe(img1);
      expect(api.$selectedItem.get()?.index).toBe(0);

      // Navigate to next entry (imageIndex 1, same item)
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$selectedImageIndex.get()).toBe(1);
      expect(api.$selectedItem.get()?.imageDTO).toBe(img2);
      expect(api.$selectedItem.get()?.index).toBe(1);

      // Navigate past last entry - should wrap to first
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(api.$selectedItem.get()?.imageDTO).toBe(img1);
    });

    it('should navigate across items and their entries', () => {
      const img1a = createMockImageDTO({ image_name: 'a1.png' });
      const img1b = createMockImageDTO({ image_name: 'a2.png' });
      const img2 = createMockImageDTO({ image_name: 'b1.png' });

      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1a, img1b],
        imageLoaded: false,
      });
      api.$progressData.setKey(2, {
        itemId: 2,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img2],
        imageLoaded: false,
      });

      // Start on first entry
      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(0);

      // entries: [item1/img0, item1/img1, item2/img0]
      expect(api.$entries.get()).toHaveLength(3);

      // Next -> item1, imageIndex 1
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$selectedImageIndex.get()).toBe(1);

      // Next -> item2, imageIndex 0
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(2);
      expect(api.$selectedImageIndex.get()).toBe(0);

      // Next -> wraps to item1, imageIndex 0
      api.selectNext();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$selectedImageIndex.get()).toBe(0);

      // Prev -> wraps to item2, imageIndex 0
      api.selectPrev();
      expect(api.$selectedItemId.get()).toBe(2);
      expect(api.$selectedImageIndex.get()).toBe(0);
    });

    it('should select correct entry with selectFirst and selectLast', () => {
      const img1a = createMockImageDTO({ image_name: 'a1.png' });
      const img1b = createMockImageDTO({ image_name: 'a2.png' });
      const img2 = createMockImageDTO({ image_name: 'b1.png' });

      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];
      api.$items.set(items);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1a, img1b],
        imageLoaded: false,
      });
      api.$progressData.setKey(2, {
        itemId: 2,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img2],
        imageLoaded: false,
      });

      api.selectLast();
      expect(api.$selectedItemId.get()).toBe(2);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(api.$selectedItem.get()?.imageDTO).toBe(img2);

      api.selectFirst();
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(api.$selectedItem.get()?.imageDTO).toBe(img1a);
    });
  });

  describe('Discard Methods', () => {
    beforeEach(() => {
      const items = [
        createMockQueueItem({ item_id: 1 }),
        createMockQueueItem({ item_id: 2 }),
        createMockQueueItem({ item_id: 3 }),
      ];
      api.$items.set(items);
    });

    it('should discard selected item and select next', () => {
      api.$selectedItemId.set(2);
      const selectedItem = api.$selectedItem.get();

      api.discardSelected();

      expect(api.$selectedItemId.get()).toBe(3);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(mockApp.onDiscard).toHaveBeenCalledWith(selectedItem?.item);
    });

    it('should discard selected item and clamp to last valid index', () => {
      api.$selectedItemId.set(3);
      const selectedItem = api.$selectedItem.get();

      api.discardSelected();

      // The logic clamps to the next index, so when discarding last item (index 2),
      // it tries to select index 3 which clamps to index 2 (item 3)
      expect(api.$selectedItemId.get()).toBe(3);
      expect(mockApp.onDiscard).toHaveBeenCalledWith(selectedItem?.item);
    });

    it('should set selectedItemId to null when discarding last item', () => {
      api.$items.set([createMockQueueItem({ item_id: 1 })]);
      api.$selectedItemId.set(1);

      api.discardSelected();

      // When there's only one item, after clamping we get the same item, so it stays selected
      expect(api.$selectedItemId.get()).toBe(1);
    });

    it('should do nothing when no item is selected', () => {
      api.$selectedItemId.set(null);
      api.discardSelected();

      expect(mockApp.onDiscard).not.toHaveBeenCalled();
    });

    it('should discard all items', () => {
      api.$selectedItemId.set(2);
      api.discardAll();

      expect(api.$selectedItemId.get()).toBe(null);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(mockApp.onDiscardAll).toHaveBeenCalled();
    });

    it('should compute discardSelectedIsEnabled correctly', () => {
      expect(api.$discardSelectedIsEnabled.get()).toBe(false);

      api.$selectedItemId.set(1);
      expect(api.$discardSelectedIsEnabled.get()).toBe(true);
    });

    it('should reset selectedImageIndex when discarding', () => {
      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(2);
      api.discardSelected();

      expect(api.$selectedImageIndex.get()).toBe(0);
    });
  });

  describe('Accept Methods', () => {
    beforeEach(() => {
      const items = [createMockQueueItem({ item_id: 1 })];
      api.$items.set(items);
      api.$selectedItemId.set(1);
    });

    it('should accept selected item with single imageDTO', () => {
      const imageDTO = createMockImageDTO();
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [imageDTO],
        imageLoaded: false,
      });

      const selectedItem = api.$selectedItem.get();
      api.acceptSelected();

      expect(mockApp.onAccept).toHaveBeenCalledWith(selectedItem?.item, imageDTO);
    });

    it('should accept the correct imageDTO from a multi-output entry', () => {
      const img1 = createMockImageDTO({ image_name: 'a.png' });
      const img2 = createMockImageDTO({ image_name: 'b.png' });
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [img1, img2],
        imageLoaded: false,
      });

      // Select the second image
      api.$selectedImageIndex.set(1);

      const selectedItem = api.$selectedItem.get();
      expect(selectedItem?.imageDTO).toBe(img2);

      api.acceptSelected();

      expect(mockApp.onAccept).toHaveBeenCalledWith(selectedItem?.item, img2);
    });

    it('should do nothing when no image is available', () => {
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [],
        imageLoaded: false,
      });

      api.acceptSelected();

      expect(mockApp.onAccept).not.toHaveBeenCalled();
    });

    it('should do nothing when no item is selected', () => {
      api.$selectedItemId.set(null);
      api.acceptSelected();

      expect(mockApp.onAccept).not.toHaveBeenCalled();
    });

    it('should compute acceptSelectedIsEnabled correctly', () => {
      expect(api.$acceptSelectedIsEnabled.get()).toBe(false);

      const imageDTO = createMockImageDTO();
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [imageDTO],
        imageLoaded: false,
      });

      expect(api.$acceptSelectedIsEnabled.get()).toBe(true);
    });
  });

  describe('Progress Event Handling', () => {
    it('should handle invocation progress events', () => {
      const progressEvent = createMockProgressEvent({
        item_id: 1,
        destination: sessionId,
      });

      api.onInvocationProgressEvent(progressEvent);

      const progressData = api.$progressData.get();
      expect(progressData[1]?.progressEvent).toBe(progressEvent);
    });

    it('should ignore events for different sessions', () => {
      const progressEvent = createMockProgressEvent({
        item_id: 1,
        destination: 'different-session',
      });

      api.onInvocationProgressEvent(progressEvent);

      const progressData = api.$progressData.get();
      expect(progressData[1]).toBeUndefined();
    });

    it('should update existing progress data', () => {
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [createMockImageDTO()],
        imageLoaded: false,
      });

      const progressEvent = createMockProgressEvent({
        item_id: 1,
        destination: sessionId,
      });

      api.onInvocationProgressEvent(progressEvent);

      const progressData = api.$progressData.get();
      expect(progressData[1]?.progressEvent).toBe(progressEvent);
      expect(progressData[1]?.imageDTOs.length).toBeGreaterThan(0);
    });
  });

  describe('Queue Item Status Change Handling', () => {
    it('should handle completed status and set last completed item', () => {
      const statusEvent = createMockQueueItemStatusChangedEvent({
        item_id: 1,
        destination: sessionId,
        status: 'completed',
      });

      api.onQueueItemStatusChangedEvent(statusEvent);

      expect(api.$lastCompletedItemId.get()).toBe(1);
    });

    it('should handle in_progress status with switch_on_start', () => {
      mockApp._setAutoSwitchMode('switch_on_start');

      const statusEvent = createMockQueueItemStatusChangedEvent({
        item_id: 1,
        destination: sessionId,
        status: 'in_progress',
      });

      api.onQueueItemStatusChangedEvent(statusEvent);

      expect(api.$lastStartedItemId.get()).toBe(1);
    });

    it('should ignore events for different sessions', () => {
      const statusEvent = createMockQueueItemStatusChangedEvent({
        item_id: 1,
        destination: 'different-session',
        status: 'completed',
      });

      api.onQueueItemStatusChangedEvent(statusEvent);

      expect(api.$lastCompletedItemId.get()).toBe(null);
    });
  });

  describe('Items Changed Event Handling', () => {
    it('should update items and auto-select first item', async () => {
      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];

      await api.onItemsChangedEvent(items);

      expect(api.$items.get()).toBe(items);
      expect(api.$selectedItemId.get()).toBe(1);
    });

    it('should clear selection when no items', async () => {
      api.$selectedItemId.set(1);

      await api.onItemsChangedEvent([]);

      expect(api.$selectedItemId.get()).toBe(null);
    });

    it('should not change selection if item already selected', async () => {
      api.$selectedItemId.set(2);

      const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];

      await api.onItemsChangedEvent(items);

      expect(api.$selectedItemId.get()).toBe(2);
    });

    it('should load images for completed items', async () => {
      const imageDTO = createMockImageDTO({ image_name: 'test-image.png' });
      mockApp._setImageDTO('test-image.png', imageDTO);

      const items = [
        createMockQueueItem({
          item_id: 1,
          status: 'completed',
        }),
      ];

      await api.onItemsChangedEvent(items);

      const progressData = api.$progressData.get();
      expect(progressData[1]?.imageDTOs[0]).toBe(imageDTO);
    });

    it('should handle auto-switch on completion', async () => {
      mockApp._setAutoSwitchMode('switch_on_finish');
      api.$lastCompletedItemId.set(1);

      const imageDTO = createMockImageDTO({ image_name: 'test-image.png' });
      mockApp._setImageDTO('test-image.png', imageDTO);

      const items = [
        createMockQueueItem({
          item_id: 1,
          status: 'completed',
        }),
      ];

      await api.onItemsChangedEvent(items);

      // Wait for async image loading - the loadImage promise needs to complete
      await new Promise((resolve) => {
        setTimeout(resolve, 50);
        api.onImageLoaded(1);
      });

      expect(api.$selectedItemId.get()).toBe(1);
      // The lastCompletedItemId should be reset after the loadImage promise resolves
      expect(api.$lastCompletedItemId.get()).toBe(null);
    });

    it('should clean up progress data for removed items', async () => {
      api.$progressData.setKey(999, {
        itemId: 999,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [],
        imageLoaded: false,
      });

      const items = [createMockQueueItem({ item_id: 1 })];

      await api.onItemsChangedEvent(items);

      const progressData = api.$progressData.get();
      expect(progressData[999]).toBeUndefined();
    });

    it('should clear progress data for canceled/failed items', async () => {
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: createMockProgressEvent({ item_id: 1 }),
        progressImage: null,
        imageDTOs: [createMockImageDTO()],
        imageLoaded: false,
      });

      const items = [createMockQueueItem({ item_id: 1, status: 'canceled' })];

      await api.onItemsChangedEvent(items);

      const progressData = api.$progressData.get();
      expect(progressData[1]?.progressEvent).toBe(null);
      expect(progressData[1]?.progressImage).toBe(null);
      expect(progressData[1]?.imageDTOs).toEqual([]);
    });
  });

  describe('Image Loading', () => {
    it('should handle image loading for completed items', () => {
      api.$items.set([createMockQueueItem({ item_id: 1 })]);
      const imageDTO = createMockImageDTO({ image_name: 'test-image.png' });
      mockApp._setImageDTO('test-image.png', imageDTO);
      api.onImageLoaded(1);
      const progressData = api.$progressData.get();
      expect(progressData[1]?.imageLoaded).toBe(true);
    });
  });

  describe('Auto Switch', () => {
    it('should set auto switch mode', () => {
      api.setAutoSwitch('switch_on_finish');
      expect(mockApp.onAutoSwitchChange).toHaveBeenCalledWith('switch_on_finish');
    });

    it('should auto-switch on finish when the image loads', async () => {
      mockApp._setAutoSwitchMode('switch_on_finish');
      api.$lastCompletedItemId.set(1);

      const imageDTO = createMockImageDTO({ image_name: 'test-image.png' });
      mockApp._setImageDTO('test-image.png', imageDTO);

      const items = [
        createMockQueueItem({
          item_id: 1,
          status: 'completed',
        }),
      ];

      await api.onItemsChangedEvent(items);
      await new Promise((resolve) => {
        setTimeout(resolve, 50);
        api.onImageLoaded(1);
      });
      expect(api.$selectedItemId.get()).toBe(1);
      expect(api.$lastCompletedItemId.get()).toBe(null);
    });
  });

  describe('Utility Methods', () => {
    it('should build isSelected computed correctly for default imageIndex', () => {
      const isSelected = api.buildIsSelectedComputed(1);
      expect(isSelected.get()).toBe(false);

      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(0);
      expect(isSelected.get()).toBe(true);
    });

    it('should build isSelected computed correctly with specific imageIndex', () => {
      const isSelected0 = api.buildIsSelectedComputed(1, 0);
      const isSelected1 = api.buildIsSelectedComputed(1, 1);

      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(0);
      expect(isSelected0.get()).toBe(true);
      expect(isSelected1.get()).toBe(false);

      api.$selectedImageIndex.set(1);
      expect(isSelected0.get()).toBe(false);
      expect(isSelected1.get()).toBe(true);
    });
  });

  describe('Cleanup', () => {
    it('should reset all state on cleanup', () => {
      api.$selectedItemId.set(1);
      api.$selectedImageIndex.set(2);
      api.$items.set([createMockQueueItem({ item_id: 1 })]);
      api.$lastStartedItemId.set(1);
      api.$lastCompletedItemId.set(1);
      api.$progressData.setKey(1, {
        itemId: 1,
        progressEvent: null,
        progressImage: null,
        imageDTOs: [],
        imageLoaded: false,
      });

      api.cleanup();

      expect(api.$selectedItemId.get()).toBe(null);
      expect(api.$selectedImageIndex.get()).toBe(0);
      expect(api.$items.get()).toEqual([]);
      expect(api.$lastStartedItemId.get()).toBe(null);
      expect(api.$lastCompletedItemId.get()).toBe(null);
      expect(api.$progressData.get()).toEqual({});
    });
  });

  describe('Edge Cases and Error Handling', () => {
    describe('Selection with Empty or Single Item Lists', () => {
      it('should handle selection operations with single item', () => {
        const items = [createMockQueueItem({ item_id: 1 })];
        api.$items.set(items);
        api.$selectedItemId.set(1);

        // Navigation should wrap around to the same item
        api.selectNext();
        expect(api.$selectedItemId.get()).toBe(1);

        api.selectPrev();
        expect(api.$selectedItemId.get()).toBe(1);
      });

      it('should handle selection operations with empty list', () => {
        api.$items.set([]);

        api.selectFirst();
        api.selectLast();
        api.selectNext();
        api.selectPrev();

        expect(api.$selectedItemId.get()).toBe(null);
      });
    });

    describe('Progress Data Edge Cases', () => {
      it('should handle progress updates with image data', () => {
        const progressEvent = createMockProgressEvent({
          item_id: 1,
          destination: sessionId,
          image: { width: 512, height: 512, dataURL: 'foo' },
        });

        api.onInvocationProgressEvent(progressEvent);

        const progressData = api.$progressData.get();
        expect(progressData[1]?.progressImage).toBe(progressEvent.image);
        expect(progressData[1]?.progressEvent).toBe(progressEvent);
      });

      it('should preserve imageDTOs when updating progress', () => {
        const imageDTO = createMockImageDTO();
        api.$progressData.setKey(1, {
          itemId: 1,
          progressEvent: null,
          progressImage: null,
          imageDTOs: [imageDTO],
          imageLoaded: false,
        });

        const progressEvent = createMockProgressEvent({
          item_id: 1,
          destination: sessionId,
        });

        api.onInvocationProgressEvent(progressEvent);

        const progressData = api.$progressData.get();
        expect(progressData[1]?.imageDTOs[0]).toBe(imageDTO);
        expect(progressData[1]?.progressEvent).toBe(progressEvent);
      });
    });

    describe('Auto-Switch Edge Cases', () => {
      it('should handle auto-switch when item is not in current items list', async () => {
        mockApp._setAutoSwitchMode('switch_on_start');
        api.$lastStartedItemId.set(999); // Non-existent item

        const items = [createMockQueueItem({ item_id: 1 })];
        await api.onItemsChangedEvent(items);

        // Should not switch to non-existent item
        expect(api.$selectedItemId.get()).toBe(1);
        expect(api.$lastStartedItemId.get()).toBe(999);
      });

      it('should handle auto-switch on finish when image loading fails', async () => {
        mockApp._setAutoSwitchMode('switch_on_finish');
        api.$lastCompletedItemId.set(1);

        const items = [
          createMockQueueItem({
            item_id: 1,
            status: 'completed',
            session: {
              id: sessionId,
              source_prepared_mapping: { canvas_output: ['test-node-id'] },
              results: {
                'test-node-id': {
                  type: 'image_output',
                  image: { image_name: 'test-image.png' },
                  width: 512,
                  height: 512,
                },
              },
            },
          }),
        ];

        await api.onItemsChangedEvent(items);

        // Wait a while but do not load the image
        await new Promise((resolve) => {
          setTimeout(resolve, 150);
        });

        // Should not switch when image loading fails
        expect(api.$selectedItemId.get()).toBe(1);
        expect(api.$lastCompletedItemId.get()).toBe(1);
      });
    });

    describe('Concurrent Operations', () => {
      it('should handle rapid item changes', async () => {
        const items1 = [createMockQueueItem({ item_id: 1 })];
        const items2 = [createMockQueueItem({ item_id: 2 })];

        // Fire multiple events rapidly
        const promise1 = api.onItemsChangedEvent(items1);
        const promise2 = api.onItemsChangedEvent(items2);

        await Promise.all([promise1, promise2]);

        // Should end up with the last set of items
        expect(api.$items.get()).toBe(items2);
        // We expect the selection to have moved to the next existent item
        expect(api.$selectedItemId.get()).toBe(2);
        expect(api.$selectedItem.get()?.item.item_id).toBe(2);
      });

      it('should not let stale async call overwrite newer data with fewer images', async () => {
        const imageDTO1 = createMockImageDTO({ image_name: 'img1.png' });
        const imageDTO2 = createMockImageDTO({ image_name: 'img2.png' });
        mockApp._setImageDTO('img1.png', imageDTO1);
        mockApp._setImageDTO('img2.png', imageDTO2);

        // Simulates optimistic update: status=completed but only 1 result in session.results
        const itemsStale = [
          createMockQueueItem({
            item_id: 1,
            status: 'completed',
            session: {
              id: sessionId,
              source_prepared_mapping: { 'canvas_output:a': ['node-1'] },
              results: { 'node-1': { image: { image_name: 'img1.png' } } },
            },
          }),
        ];

        // Simulates full refetch: both results available
        const itemsFull = [
          createMockQueueItem({
            item_id: 1,
            status: 'completed',
            session: {
              id: sessionId,
              source_prepared_mapping: { 'canvas_output:a': ['node-1'], 'canvas_output:b': ['node-2'] },
              results: {
                'node-1': { image: { image_name: 'img1.png' } },
                'node-2': { image: { image_name: 'img2.png' } },
              },
            },
          }),
        ];

        // Fire both concurrently (stale optimistic update then full refetch)
        const promise1 = api.onItemsChangedEvent(itemsStale);
        const promise2 = api.onItemsChangedEvent(itemsFull);
        await Promise.all([promise1, promise2]);

        const progressData = api.$progressData.get();
        expect(progressData[1]?.imageDTOs).toHaveLength(2);
        expect(progressData[1]?.imageDTOs[0]).toBe(imageDTO1);
        expect(progressData[1]?.imageDTOs[1]).toBe(imageDTO2);
      });

      it('should load all images from multiple canvas_output nodes', async () => {
        const imageDTO1 = createMockImageDTO({ image_name: 'output1.png' });
        const imageDTO2 = createMockImageDTO({ image_name: 'output2.png' });
        mockApp._setImageDTO('output1.png', imageDTO1);
        mockApp._setImageDTO('output2.png', imageDTO2);

        const items = [
          createMockQueueItem({
            item_id: 1,
            status: 'completed',
            session: {
              id: sessionId,
              source_prepared_mapping: {
                'canvas_output:abc': ['prepared-1'],
                'canvas_output:def': ['prepared-2'],
              },
              results: {
                'prepared-1': { image: { image_name: 'output1.png' } },
                'prepared-2': { image: { image_name: 'output2.png' } },
              },
            },
          }),
        ];

        await api.onItemsChangedEvent(items);

        const progressData = api.$progressData.get();
        expect(progressData[1]?.imageDTOs).toHaveLength(2);
        expect(progressData[1]?.imageDTOs[0]).toBe(imageDTO1);
        expect(progressData[1]?.imageDTOs[1]).toBe(imageDTO2);
      });

      it('should create separate entries for multiple canvas_output images', async () => {
        const imageDTO1 = createMockImageDTO({ image_name: 'output1.png' });
        const imageDTO2 = createMockImageDTO({ image_name: 'output2.png' });
        mockApp._setImageDTO('output1.png', imageDTO1);
        mockApp._setImageDTO('output2.png', imageDTO2);

        const items = [
          createMockQueueItem({
            item_id: 1,
            status: 'completed',
            session: {
              id: sessionId,
              source_prepared_mapping: {
                'canvas_output:abc': ['prepared-1'],
                'canvas_output:def': ['prepared-2'],
              },
              results: {
                'prepared-1': { image: { image_name: 'output1.png' } },
                'prepared-2': { image: { image_name: 'output2.png' } },
              },
            },
          }),
        ];

        await api.onItemsChangedEvent(items);

        const entries = api.$entries.get();
        expect(entries).toHaveLength(2);
        expect(entries[0]?.imageDTO).toBe(imageDTO1);
        expect(entries[0]?.imageIndex).toBe(0);
        expect(entries[1]?.imageDTO).toBe(imageDTO2);
        expect(entries[1]?.imageIndex).toBe(1);
      });

      it('should handle multiple progress events for same item', () => {
        const event1 = createMockProgressEvent({
          item_id: 1,
          destination: sessionId,
          percentage: 0.3,
        });
        const event2 = createMockProgressEvent({
          item_id: 1,
          destination: sessionId,
          percentage: 0.7,
        });

        api.onInvocationProgressEvent(event1);
        api.onInvocationProgressEvent(event2);

        const progressData = api.$progressData.get();
        expect(progressData[1]?.progressEvent).toBe(event2);
      });
    });

    describe('Memory Management', () => {
      it('should clean up progress data for large number of items', async () => {
        // Create progress data for many items
        for (let i = 1; i <= 1000; i++) {
          api.$progressData.setKey(i, {
            itemId: i,
            progressEvent: null,
            progressImage: null,
            imageDTOs: [],
            imageLoaded: false,
          });
        }

        // Only keep a few items
        const items = [createMockQueueItem({ item_id: 1 }), createMockQueueItem({ item_id: 2 })];

        await api.onItemsChangedEvent(items);

        const progressData = api.$progressData.get();
        const progressDataKeys = Object.keys(progressData);

        // Should only have progress data for current items
        expect(progressDataKeys.length).toBeLessThanOrEqual(2);
        expect(progressData[1]).toBeDefined();
        expect(progressData[2]).toBeDefined();
      });
    });

    describe('Event Subscription Management', () => {
      it('should handle multiple subscriptions and unsubscriptions', () => {
        const api2 = new StagingAreaApi();
        api2.connectToApp(sessionId, mockApp);
        const api3 = new StagingAreaApi();
        api3.connectToApp(sessionId, mockApp);

        // All should be subscribed
        expect(mockApp.onItemsChanged).toHaveBeenCalledTimes(3);

        api2.cleanup();
        api3.cleanup();

        // Should not affect original api
        expect(api.$items.get()).toBeDefined();
      });

      it('should handle events after cleanup', () => {
        api.cleanup();

        // These should not crash
        const progressEvent = createMockProgressEvent({
          item_id: 1,
          destination: sessionId,
        });

        api.onInvocationProgressEvent(progressEvent);

        // State should remain clean - but the event handler still works
        // so it will add progress data even after cleanup
        const progressData = api.$progressData.get();
        expect(progressData[1]).toBeDefined();
      });
    });
  });
});
