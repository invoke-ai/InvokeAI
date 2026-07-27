import { ErrResult, OkResult } from 'common/util/result';
import {
  $splatOverlay,
  applyConvertTo3DResult,
  updateSplatOverlayRect,
} from 'features/controlLayers/components/SplatOverlay/state';
import { toast } from 'features/toast/toast';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('features/toast/toast', () => ({
  toast: vi.fn(),
}));

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    trace: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

vi.mock('i18next', () => {
  const t = (key: string) => key;
  return { t, default: { t } };
});

const rect = { x: 10, y: 20, width: 100, height: 50 };

describe('applyConvertTo3DResult', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    $splatOverlay.set(null);
  });

  it('promotes the current loading session to ready, preserving the latest rect', () => {
    $splatOverlay.set({ status: 'loading', sessionId: 'session-1', rect });
    // The frame is movable while loading — the ready state must use the moved rect, not the initial one.
    const movedRect = { ...rect, x: 42 };
    updateSplatOverlayRect(movedRect);

    applyConvertTo3DResult(OkResult('https://example.com/asset.ply'), 'session-1');

    expect($splatOverlay.get()).toEqual({
      status: 'ready',
      sessionId: 'session-1',
      assetUrl: 'https://example.com/asset.ply',
      rect: movedRect,
    });
    expect(toast).not.toHaveBeenCalled();
  });

  it('shows an error toast and closes the overlay when the current session fails', () => {
    $splatOverlay.set({ status: 'loading', sessionId: 'session-1', rect });

    applyConvertTo3DResult(ErrResult(new Error('CUDA out of memory')), 'session-1');

    expect($splatOverlay.get()).toBeNull();
    expect(toast).toHaveBeenCalledWith({ status: 'error', title: 'controlLayers.convertTo3D.generationError' });
  });

  it('stays silent when a failure arrives after the user cancelled (overlay closed)', () => {
    // Cancelling closes the overlay and aborts the run; the aborted run's rejection then lands here.
    applyConvertTo3DResult(ErrResult(new Error('aborted')), 'session-1');

    expect($splatOverlay.get()).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it('stays silent and leaves state untouched when a failure arrives for a replaced session', () => {
    const newSession = { status: 'loading', sessionId: 'session-2', rect } as const;
    $splatOverlay.set(newSession);

    applyConvertTo3DResult(ErrResult(new Error('aborted')), 'session-1');

    expect($splatOverlay.get()).toEqual(newSession);
    expect(toast).not.toHaveBeenCalled();
  });

  it('discards a success that arrives for a replaced session', () => {
    const newSession = { status: 'loading', sessionId: 'session-2', rect } as const;
    $splatOverlay.set(newSession);

    applyConvertTo3DResult(OkResult('https://example.com/stale.ply'), 'session-1');

    expect($splatOverlay.get()).toEqual(newSession);
    expect(toast).not.toHaveBeenCalled();
  });

  it('discards a success that arrives after cancel', () => {
    applyConvertTo3DResult(OkResult('https://example.com/stale.ply'), 'session-1');

    expect($splatOverlay.get()).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });
});
