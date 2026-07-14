import { describe, expect, it, vi } from 'vitest';

import { createCanvasOperationStores } from './operationStores';

describe('createCanvasOperationStores', () => {
  it('publishes filter and Select Object session snapshots independently', () => {
    const stores = createCanvasOperationStores();
    const onFilter = vi.fn();
    const onSam = vi.fn();
    stores.filterSession.subscribe(onFilter);
    stores.samSession.subscribe(onSam);

    stores.filterSession.set({ status: 'ready' } as never);
    expect(onFilter).toHaveBeenCalledOnce();
    expect(onSam).not.toHaveBeenCalled();

    stores.samSession.set({ status: 'ready' } as never);
    expect(onSam).toHaveBeenCalledOnce();
  });

  it('does not notify for an identical snapshot reference', () => {
    const stores = createCanvasOperationStores();
    const listener = vi.fn();
    const snapshot = { status: 'ready' } as never;
    stores.filterSession.subscribe(listener);
    stores.filterSession.set(snapshot);
    stores.filterSession.set(snapshot);
    expect(listener).toHaveBeenCalledOnce();
  });
});
