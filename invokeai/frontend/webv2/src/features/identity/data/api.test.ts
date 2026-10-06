import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn(), getHttpAuthToken: vi.fn() }));

vi.mock('@platform/transport/http', () => ({
  apiFetch: mocks.apiFetch,
  apiFetchJson: mocks.apiFetchJson,
  getHttpAuthToken: mocks.getHttpAuthToken,
}));

import { isPasswordChangePending } from '@features/identity/core/tokenStorage';

import { createUser, deleteUser, updateUser } from './api';

describe('Identity user mutations', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset().mockResolvedValue(new Response());
    mocks.apiFetchJson.mockReset().mockResolvedValue({});
    mocks.getHttpAuthToken.mockReset().mockReturnValue(null);
  });

  it('creates users with the backend DTO shape', async () => {
    const request = { display_name: 'Ada', email: 'ada@example.com', is_admin: true, password: 'secret' };

    await createUser(request);

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/auth/users', {
      body: JSON.stringify(request),
      method: 'POST',
    });
  });

  it('encodes user ids for update and delete mutations', async () => {
    await updateUser('user/with space', { is_active: false });
    await deleteUser('user/with space');

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/auth/users/user%2Fwith%20space', {
      body: JSON.stringify({ is_active: false }),
      method: 'PATCH',
    });
    expect(mocks.apiFetch).toHaveBeenCalledWith('/api/v1/auth/users/user%2Fwith%20space', { method: 'DELETE' });
  });

  it('marks only an admin self-password reset while its request is in flight', async () => {
    const token = `header.${btoa(JSON.stringify({ token_epoch: 3, user_id: 'admin' }))}.signature`;
    mocks.getHttpAuthToken.mockReturnValue(token);
    let finish!: (value: object) => void;
    mocks.apiFetchJson.mockImplementationOnce(
      () =>
        new Promise<object>((resolve) => {
          finish = resolve;
        })
    );

    const selfReset = updateUser('admin', { password: 'new' });
    expect(isPasswordChangePending(token)).toBe(true);
    finish({ user_id: 'admin' });
    await selfReset;
    expect(isPasswordChangePending(token)).toBe(false);

    await updateUser('other-user', { password: 'new' });
    expect(isPasswordChangePending(token)).toBe(false);
  });
});
