import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', () => ({ apiFetch: mocks.apiFetch, apiFetchJson: mocks.apiFetchJson }));

import { createUser, deleteUser, updateUser } from './api';

describe('Identity user mutations', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset().mockResolvedValue(new Response());
    mocks.apiFetchJson.mockReset().mockResolvedValue({});
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
});
