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

  it('replaces duplicate global command registrations', async () => {
    const firstHandler = vi.fn(() => 'first');
    const secondHandler = vi.fn(() => 'second');
    const disposeFirst = commandApi.register({ handler: firstHandler, id: 'test.duplicate-command', title: 'First' });
    const disposeSecond = commandApi.register({
      handler: secondHandler,
      id: 'test.duplicate-command',
      title: 'Second',
    });

    await expect(commandApi.execute('test.duplicate-command')).resolves.toBe('second');

    disposeFirst();

    await expect(commandApi.execute('test.duplicate-command')).resolves.toBe('second');

    disposeSecond();

    await expect(commandApi.execute('test.duplicate-command')).resolves.toBeUndefined();
  });

  it('replaces duplicate command registrations from the same source', async () => {
    const firstHandler = vi.fn(() => 'first');
    const secondHandler = vi.fn(() => 'second');
    const source = { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' } as const;
    const disposeFirst = commandApi.register({
      handler: firstHandler,
      id: 'test.replace-command',
      source,
      title: 'First',
    });
    const disposeSecond = commandApi.register({
      handler: secondHandler,
      id: 'test.replace-command',
      source,
      title: 'Second',
    });

    await expect(commandApi.executeForSource('test.replace-command', source)).resolves.toBe('second');

    disposeFirst();

    await expect(commandApi.executeForSource('test.replace-command', source)).resolves.toBe('second');

    disposeSecond();

    await expect(commandApi.executeForSource('test.replace-command', source)).resolves.toBeUndefined();
  });

  it('does not execute a command registered by the same widget source in another project', async () => {
    const handler = vi.fn(() => 'project-1');
    const dispose = commandApi.register({
      handler,
      id: 'test.project-command',
      source: { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' },
      title: 'Project command',
    });

    await expect(
      commandApi.executeForSource('test.project-command', {
        instanceId: 'alpha',
        projectId: 'project-2',
        region: 'right',
        typeId: 'test-widget',
      })
    ).resolves.toBeUndefined();
    expect(handler).not.toHaveBeenCalled();

    dispose();
  });

  it('does not fall back to another widget instance when executing for a source', async () => {
    const handler = vi.fn(() => 'beta');
    const dispose = commandApi.register({
      handler,
      id: 'test.scoped-command',
      source: { instanceId: 'beta', projectId: 'project-1', region: 'right', typeId: 'test-widget' },
      title: 'Scoped command',
    });

    await expect(
      commandApi.executeForSource('test.scoped-command', {
        instanceId: 'alpha',
        projectId: 'project-1',
        region: 'right',
        typeId: 'test-widget',
      })
    ).resolves.toBeUndefined();
    expect(handler).not.toHaveBeenCalled();

    dispose();
  });
});
