import { toast } from 'features/toast/toast';
import { describe, expect, it, vi } from 'vitest';

import { addBulkDownloadListeners } from './bulkDownload';

vi.mock('features/toast/toast', () => ({ toast: vi.fn() }));
vi.mock('i18next', () => ({ t: vi.fn((key: string) => key) }));

/** Collects the effects the listeners register, in registration order. */
const collectEffects = () => {
  const effects: ((action: { payload: unknown }) => void)[] = [];
  const startAppListening = vi.fn(({ effect }: { effect: (action: { payload: unknown }) => void }) => {
    effects.push(effect);
  });
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  addBulkDownloadListeners(startAppListening as any);
  return effects;
};

describe('bulk download listeners', () => {
  it('raises no toast for a fulfilled action with no payload', () => {
    // `fetchBaseQuery` resolves an empty response entity as `data: null`, so a proxy that
    // strips the body off the 202 fulfils this action with nothing in it. Dereferencing the
    // payload would throw — but raising the toast anyway is no better: it is persistent
    // (`duration: null`) and is dismissed by name when the zip lands, so without a name it
    // gets a random id that the socket handler's close call can never match, and the banner
    // stays on screen forever. The download is unaffected; its completion toast still arrives.
    const [onFulfilled] = collectEffects();

    expect(() => onFulfilled?.({ payload: null })).not.toThrow();
    expect(toast).not.toHaveBeenCalled();
  });

  it('keys the preparing toast on the item name when there is one', () => {
    // Distinct ids matter: the background task can finish in under 20ms, so the "ready"
    // toast may already be on screen when this one is raised.
    const [onFulfilled] = collectEffects();

    onFulfilled?.({ payload: { bulk_download_item_name: 'item-1.zip', response: 'on its way' } });

    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'preparing:item-1.zip', description: 'on its way' })
    );
  });
});
