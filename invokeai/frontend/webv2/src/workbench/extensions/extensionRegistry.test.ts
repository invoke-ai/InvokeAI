import { describe, expect, it, vi } from 'vitest';

import { createExtensionRegistry } from './extensionRegistry';

describe('createExtensionRegistry', () => {
  it('registers, executes, and disposes commands by id', async () => {
    const { commands } = createExtensionRegistry();
    const handler = vi.fn(() => 'ok');
    const dispose = commands.register({ handler, id: 'test.command', title: 'Test command' });

    await expect(commands.execute('test.command', 'arg')).resolves.toBe('ok');
    expect(handler).toHaveBeenCalledWith('arg');

    dispose();

    await expect(commands.execute('test.command')).resolves.toBeUndefined();
  });

  it('replaces duplicate global command registrations', async () => {
    const { commands } = createExtensionRegistry();
    const firstHandler = vi.fn(() => 'first');
    const secondHandler = vi.fn(() => 'second');
    const disposeFirst = commands.register({ handler: firstHandler, id: 'test.duplicate-command', title: 'First' });
    const disposeSecond = commands.register({
      handler: secondHandler,
      id: 'test.duplicate-command',
      title: 'Second',
    });

    await expect(commands.execute('test.duplicate-command')).resolves.toBe('second');

    disposeFirst();

    await expect(commands.execute('test.duplicate-command')).resolves.toBe('second');

    disposeSecond();

    await expect(commands.execute('test.duplicate-command')).resolves.toBeUndefined();
  });

  it('replaces duplicate command registrations from the same source', async () => {
    const { commands } = createExtensionRegistry();
    const firstHandler = vi.fn(() => 'first');
    const secondHandler = vi.fn(() => 'second');
    const source = { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' } as const;
    const disposeFirst = commands.register({
      handler: firstHandler,
      id: 'test.replace-command',
      source,
      title: 'First',
    });
    const disposeSecond = commands.register({
      handler: secondHandler,
      id: 'test.replace-command',
      source,
      title: 'Second',
    });

    await expect(commands.executeForSource('test.replace-command', source)).resolves.toBe('second');

    disposeFirst();

    await expect(commands.executeForSource('test.replace-command', source)).resolves.toBe('second');

    disposeSecond();

    await expect(commands.executeForSource('test.replace-command', source)).resolves.toBeUndefined();
  });

  it('does not execute a command registered by the same widget source in another project', async () => {
    const { commands } = createExtensionRegistry();
    const handler = vi.fn(() => 'project-1');
    const dispose = commands.register({
      handler,
      id: 'test.project-command',
      source: { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' },
      title: 'Project command',
    });

    await expect(
      commands.executeForSource('test.project-command', {
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
    const { commands } = createExtensionRegistry();
    const handler = vi.fn(() => 'beta');
    const dispose = commands.register({
      handler,
      id: 'test.scoped-command',
      source: { instanceId: 'beta', projectId: 'project-1', region: 'right', typeId: 'test-widget' },
      title: 'Scoped command',
    });

    await expect(
      commands.executeForSource('test.scoped-command', {
        instanceId: 'alpha',
        projectId: 'project-1',
        region: 'right',
        typeId: 'test-widget',
      })
    ).resolves.toBeUndefined();
    expect(handler).not.toHaveBeenCalled();

    dispose();
  });

  it('keeps contributions isolated between registries', async () => {
    const first = createExtensionRegistry();
    const second = createExtensionRegistry();
    const handler = vi.fn(() => 'first-registry');

    first.commands.register({ handler, id: 'test.isolated-command', title: 'Isolated' });
    first.palette.register({ commandId: 'test.isolated-command', title: 'Isolated' });

    await expect(second.commands.execute('test.isolated-command')).resolves.toBeUndefined();
    expect(second.stores.palette.list()).toHaveLength(0);
    expect(first.stores.palette.list()).toHaveLength(1);
  });
});
