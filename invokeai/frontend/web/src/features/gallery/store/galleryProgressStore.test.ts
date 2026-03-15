import { beforeEach, describe, expect, it } from 'vitest';

import {
  $galleryProgressItems,
  addGalleryProgressItems,
  clearGalleryProgressItems,
  removeGalleryProgressItem,
  selectFilteredGalleryProgressItems,
  setGalleryProgressItemInProgress,
  updateGalleryProgressItemProgress,
} from './galleryProgressStore';

describe('galleryProgressStore', () => {
  beforeEach(() => {
    clearGalleryProgressItems();
  });

  describe('addGalleryProgressItems', () => {
    it('should add items with pending status', () => {
      addGalleryProgressItems([1, 2, 3], 'batch-1', 'none');
      const items = $galleryProgressItems.get();
      expect(Object.keys(items)).toHaveLength(3);
      expect(items[1]?.status).toBe('pending');
      expect(items[1]?.batchId).toBe('batch-1');
      expect(items[1]?.targetBoardId).toBe('none');
      expect(items[1]?.progressImage).toBeNull();
    });

    it('should preserve existing items when adding new ones', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      addGalleryProgressItems([2], 'batch-2', 'board-1');
      const items = $galleryProgressItems.get();
      expect(Object.keys(items)).toHaveLength(2);
      expect(items[1]?.batchId).toBe('batch-1');
      expect(items[2]?.batchId).toBe('batch-2');
    });
  });

  describe('setGalleryProgressItemInProgress', () => {
    it('should update status to in_progress', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      setGalleryProgressItemInProgress(1);
      expect($galleryProgressItems.get()[1]?.status).toBe('in_progress');
    });

    it('should no-op for non-existent items', () => {
      setGalleryProgressItemInProgress(999);
      expect(Object.keys($galleryProgressItems.get())).toHaveLength(0);
    });
  });

  describe('updateGalleryProgressItemProgress', () => {
    it('should update progress image and percentage', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      const progressImage = { dataURL: 'data:image/png;base64,abc', width: 512, height: 512 };
      updateGalleryProgressItemProgress(1, progressImage, 0.5, 'Denoising');
      const item = $galleryProgressItems.get()[1];
      expect(item?.progressImage).toEqual(progressImage);
      expect(item?.percentage).toBe(0.5);
      expect(item?.message).toBe('Denoising');
    });

    it('should no-op for non-existent items', () => {
      updateGalleryProgressItemProgress(999, null, null, null);
      expect(Object.keys($galleryProgressItems.get())).toHaveLength(0);
    });
  });

  describe('removeGalleryProgressItem', () => {
    it('should remove the item', () => {
      addGalleryProgressItems([1, 2], 'batch-1', 'none');
      removeGalleryProgressItem(1);
      const items = $galleryProgressItems.get();
      expect(Object.keys(items)).toHaveLength(1);
      expect(items[1]).toBeUndefined();
      expect(items[2]).toBeDefined();
    });

    it('should no-op for non-existent items', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      removeGalleryProgressItem(999);
      expect(Object.keys($galleryProgressItems.get())).toHaveLength(1);
    });
  });

  describe('clearGalleryProgressItems', () => {
    it('should clear all items', () => {
      addGalleryProgressItems([1, 2, 3], 'batch-1', 'none');
      clearGalleryProgressItems();
      expect(Object.keys($galleryProgressItems.get())).toHaveLength(0);
    });
  });

  describe('selectFilteredGalleryProgressItems', () => {
    it('should filter by board ID', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      addGalleryProgressItems([2], 'batch-2', 'board-1');
      addGalleryProgressItems([3], 'batch-3', 'none');
      const items = $galleryProgressItems.get();
      const filtered = selectFilteredGalleryProgressItems(items, 'none');
      expect(filtered).toHaveLength(2);
      expect(filtered.map((i) => i.itemId)).toEqual([1, 3]);
    });

    it('should sort by enqueuedAt ascending', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      addGalleryProgressItems([2], 'batch-2', 'none');
      const items = $galleryProgressItems.get();
      const filtered = selectFilteredGalleryProgressItems(items, 'none');
      expect(filtered[0]?.itemId).toBe(1);
      expect(filtered[1]?.itemId).toBe(2);
    });

    it('should return empty array when no items match', () => {
      addGalleryProgressItems([1], 'batch-1', 'none');
      const items = $galleryProgressItems.get();
      const filtered = selectFilteredGalleryProgressItems(items, 'board-xyz');
      expect(filtered).toHaveLength(0);
    });
  });
});
