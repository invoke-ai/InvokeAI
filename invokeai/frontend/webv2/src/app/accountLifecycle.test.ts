import { beforeEach, describe, expect, it, vi } from 'vitest';

const collaborators = vi.hoisted(() => ({
  configureIdentityAccountLifecycle: vi.fn(),
  disconnectSocket: vi.fn(),
}));

vi.mock('@features/identity', () => ({
  configureIdentityAccountLifecycle: collaborators.configureIdentityAccountLifecycle,
}));

vi.mock('@platform/transport/socketHub', () => ({
  socketHub: { disconnect: collaborators.disconnectSocket },
}));

beforeEach(() => {
  vi.resetModules();
  collaborators.configureIdentityAccountLifecycle.mockReset();
  collaborators.disconnectSocket.mockReset();
});

describe('App account lifecycle composition', () => {
  it('clears the shared QueryClient synchronously on account invalidation', async () => {
    const [{ configureAppAccountLifecycle }, { queryClient }, { accountLifecycle }] = await Promise.all([
      import('./accountLifecycle'),
      import('@platform/query/client'),
      import('@platform/state/accountLifecycle'),
    ]);

    configureAppAccountLifecycle();
    accountLifecycle.activate('user-a', ':user:user-a');
    collaborators.disconnectSocket.mockClear();
    queryClient.setQueryData(['account-owned', 'probe'], { privateValue: 'user-a' });
    expect(queryClient.getQueryData(['account-owned', 'probe'])).toEqual({ privateValue: 'user-a' });

    accountLifecycle.invalidate();

    expect(queryClient.getQueryData(['account-owned', 'probe'])).toBeUndefined();
    expect(collaborators.disconnectSocket).toHaveBeenCalledOnce();
    expect(collaborators.configureIdentityAccountLifecycle).toHaveBeenCalledWith(accountLifecycle);
  });
});
