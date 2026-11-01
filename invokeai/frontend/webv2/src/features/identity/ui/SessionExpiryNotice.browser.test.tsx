import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SessionExpiryNotice } from './SessionExpiryNotice';

const harness = vi.hoisted(() => ({
  createActionToast: vi.fn(),
  onWarning: null as null | ((warning: { expiresAt: number; signal: AbortSignal }) => void),
  stop: vi.fn(),
}));

vi.mock('@features/identity/session', () => ({
  watchSessionExpiry: (onWarning: (warning: { expiresAt: number; signal: AbortSignal }) => void) => {
    harness.onWarning = onWarning;
    return harness.stop;
  },
}));
vi.mock('@platform/ui/toaster', () => ({ createActionToast: harness.createActionToast }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  harness.createActionToast.mockReset();
  harness.stop.mockReset();
  harness.onWarning = null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  host.remove();
});

describe('session expiry notice', () => {
  it('shows a warning that closes when the watcher withdraws it, and stops watching on unmount', async () => {
    await act(() => root.render(<SessionExpiryNotice />));
    expect(harness.createActionToast).not.toHaveBeenCalled();

    const controller = new AbortController();
    harness.onWarning?.({ expiresAt: Date.parse('2026-10-29T12:05:00'), signal: controller.signal });

    expect(harness.createActionToast).toHaveBeenCalledOnce();
    const [toast] = harness.createActionToast.mock.calls[0]!;
    // No action: nothing renews a session but a change the user makes (see the identity README).
    expect(toast).toMatchObject({ actions: [], duration: Number.POSITIVE_INFINITY, type: 'warning' });
    expect(toast.signal).toBe(controller.signal);

    await act(() => root.unmount());
    expect(harness.stop).toHaveBeenCalledOnce();
  });
});
