import { beforeEach, describe, expect, it } from 'vitest';

import { itemProgressStore } from './itemProgressStore';

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
