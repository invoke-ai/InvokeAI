import { beforeEach, describe, expect, it } from 'vitest';

import { getActiveProgressItemIds, itemProgressStore } from './itemProgressStore';

describe('itemProgressStore', () => {
  beforeEach(() => {
    itemProgressStore.clearAll();
  });

  it('preserves the previous live preview image when a progress update has no image payload', () => {
    itemProgressStore.set(42, {
      image: { dataUrl: 'data:image/png;base64,step-1', height: 64, width: 64 },
      message: 'Denoising 1/20',
      percentage: 0.05,
    });

    itemProgressStore.set(42, {
      message: 'Denoising 2/20',
      percentage: 0.1,
    });

    expect(itemProgressStore.get(42)).toEqual({
      image: { dataUrl: 'data:image/png;base64,step-1', height: 64, width: 64 },
      message: 'Denoising 2/20',
      percentage: 0.1,
    });
  });

  it('clears the live preview image when a progress update explicitly sets image to null', () => {
    itemProgressStore.set(42, {
      image: { dataUrl: 'data:image/png;base64,step-1', height: 64, width: 64 },
      message: 'Denoising 1/20',
      percentage: 0.05,
    });

    itemProgressStore.set(42, {
      image: null,
      message: '',
      percentage: null,
    });

    expect(itemProgressStore.get(42)).toEqual({
      image: null,
      message: '',
      percentage: null,
    });
  });
});

describe('itemProgressStore multi-session tracking', () => {
  beforeEach(() => {
    itemProgressStore.clearAll();
  });

  it('tracks concurrent sessions independently, keyed by backend item id', () => {
    // Multi-GPU runs one session per GPU, so two items report progress at once.
    itemProgressStore.set(10, { device: 'cuda:0', message: 'Denoising 1/20', percentage: 0.05 });
    itemProgressStore.set(11, { device: 'cuda:1', message: 'Denoising 3/20', percentage: 0.15 });

    expect(itemProgressStore.get(10)?.device).toBe('cuda:0');
    expect(itemProgressStore.get(11)?.device).toBe('cuda:1');
    expect(getActiveProgressItemIds()).toEqual([10, 11]);
  });

  it('orders active ids ascending regardless of which session reported first', () => {
    itemProgressStore.set(21, { message: '', percentage: null });
    itemProgressStore.set(20, { message: '', percentage: null });
    itemProgressStore.set(22, { message: '', percentage: null });

    expect(getActiveProgressItemIds()).toEqual([20, 21, 22]);
  });

  it('drops an id when its session settles, leaving the others running', () => {
    itemProgressStore.set(10, { message: '', percentage: null });
    itemProgressStore.set(11, { message: '', percentage: null });

    itemProgressStore.clear(10);

    expect(getActiveProgressItemIds()).toEqual([11]);
    expect(itemProgressStore.get(10)).toBeNull();
  });

  it('keeps the active id list stable across repeated progress frames', () => {
    itemProgressStore.set(10, { message: 'step 1', percentage: 0.1 });
    const first = getActiveProgressItemIds();

    itemProgressStore.set(10, { message: 'step 2', percentage: 0.2 });

    // Identity must survive a same-id update, or every consumer re-renders per frame.
    expect(getActiveProgressItemIds()).toBe(first);
  });

  it('empties the active list on clearAll', () => {
    itemProgressStore.set(10, { message: '', percentage: null });
    itemProgressStore.clearAll();

    expect(getActiveProgressItemIds()).toEqual([]);
  });
});
