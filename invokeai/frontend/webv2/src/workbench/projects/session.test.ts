import { beforeEach, describe, expect, it, vi } from 'vitest';

import { getClientStateValue } from './api';
import { fetchSessionBlobStrict, peekOpenProjectIds } from './session';

vi.mock('./api', () => ({
  getClientStateValue: vi.fn(),
  setClientStateValue: vi.fn(),
}));

describe('fetchSessionBlobStrict', () => {
  beforeEach(() => vi.mocked(getClientStateValue).mockReset());

  it('propagates transport failures instead of treating them as a missing session', async () => {
    const failure = new Error('backend unavailable');
    vi.mocked(getClientStateValue).mockRejectedValueOnce(failure);

    await expect(fetchSessionBlobStrict()).rejects.toBe(failure);
  });
});

describe('peekOpenProjectIds', () => {
  beforeEach(() => vi.mocked(getClientStateValue).mockReset());

  it('keeps the editor route reachable when durable queue work survives an empty session', async () => {
    vi.mocked(getClientStateValue).mockResolvedValueOnce(
      JSON.stringify({ account: {}, activeProjectId: '', openProjectIds: [] })
    );

    await expect(
      peekOpenProjectIds({
        listDurableRecoveryProjectIds: () => Promise.resolve({ kind: 'available', projectIds: ['queue-project'] }),
      })
    ).resolves.toEqual(['queue-project']);
  });

  it('keeps the editor route reachable when browser recovery storage cannot be inspected', async () => {
    vi.mocked(getClientStateValue).mockResolvedValueOnce(
      JSON.stringify({ account: {}, activeProjectId: '', openProjectIds: [] })
    );

    await expect(
      peekOpenProjectIds({
        listDurableRecoveryProjectIds: () => Promise.resolve({ kind: 'unavailable' }),
      })
    ).resolves.toBeNull();
  });

  it('does not inspect browser recovery storage for a nonempty session', async () => {
    vi.mocked(getClientStateValue).mockResolvedValueOnce(
      JSON.stringify({ account: {}, activeProjectId: 'server-project', openProjectIds: ['server-project'] })
    );
    const listDurableRecoveryProjectIds = vi.fn();

    await expect(peekOpenProjectIds({ listDurableRecoveryProjectIds })).resolves.toEqual(['server-project']);
    expect(listDurableRecoveryProjectIds).not.toHaveBeenCalled();
  });
});
