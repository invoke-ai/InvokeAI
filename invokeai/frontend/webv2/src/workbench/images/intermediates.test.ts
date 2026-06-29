import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@workbench/backend/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
}));

describe('intermediates API helpers', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('reads the intermediate image count from the images endpoint', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce(7);
    const { getIntermediatesCount } = await import('./intermediates');

    await expect(getIntermediatesCount()).resolves.toBe(7);

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/images/intermediates');
  });

  it('clears intermediate images with DELETE and returns the cleared count', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce(3);
    const { clearIntermediates } = await import('./intermediates');

    await expect(clearIntermediates()).resolves.toBe(3);

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/images/intermediates', { method: 'DELETE' });
  });
});

describe('clear intermediates UI state', () => {
  it('requires permission, a loaded nonzero count, and no active queue work', async () => {
    const { getClearIntermediatesState } = await import('./intermediates');

    expect(
      getClearIntermediatesState({ canClearIntermediates: false, hasActiveQueueWork: false, intermediatesCount: 4 })
    ).toEqual({ disabled: true, reason: 'Only administrators can clear intermediates.' });

    expect(
      getClearIntermediatesState({ canClearIntermediates: true, hasActiveQueueWork: true, intermediatesCount: 4 })
    ).toEqual({ disabled: true, reason: 'Wait for pending or running queue work to finish.' });

    expect(
      getClearIntermediatesState({ canClearIntermediates: true, hasActiveQueueWork: false, intermediatesCount: 0 })
    ).toEqual({ disabled: true, reason: 'There are no intermediates to clear.' });

    expect(
      getClearIntermediatesState({ canClearIntermediates: true, hasActiveQueueWork: false, intermediatesCount: null })
    ).toEqual({ disabled: true, reason: 'Loading intermediate count.' });

    expect(
      getClearIntermediatesState({ canClearIntermediates: true, hasActiveQueueWork: false, intermediatesCount: 4 })
    ).toEqual({ disabled: false });
  });

  it('describes the destructive clear action before confirmation', async () => {
    const { getClearIntermediatesConfirmation } = await import('./intermediates');

    expect(getClearIntermediatesConfirmation(5)).toEqual({
      body: 'This permanently deletes 5 temporary intermediate images and clears canvas layers or staged images that may reference them. Final gallery images are not removed.',
      confirmLabel: 'Clear intermediates',
      title: 'Clear intermediate images?',
    });
  });
});
