import { describe, expect, it, vi } from 'vitest';

import { commandApi } from './extensionApi';

describe('commandApi', () => {
  it('registers, executes, and disposes commands by id', async () => {
    const handler = vi.fn(() => 'ok');
    const dispose = commandApi.register({ handler, id: 'test.command', title: 'Test command' });

    await expect(commandApi.execute('test.command', 'arg')).resolves.toBe('ok');
    expect(handler).toHaveBeenCalledWith('arg');

    dispose();

    await expect(commandApi.execute('test.command')).resolves.toBeUndefined();
  });
});
