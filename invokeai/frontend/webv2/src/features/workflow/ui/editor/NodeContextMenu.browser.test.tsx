import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { NodeContextMenu, type NodeContextMenuState } from './NodeContextMenu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The App draws real keycaps; this names the command each hint is asked for. */
const adapter = {
  CommandShortcut: ({ commandId }: { commandId: string }) => <kbd>{commandId}</kbd>,
} as unknown as WorkflowUiAdapter;

describe('NodeContextMenu', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const menuStates: Record<'no' | 'yes', NodeContextMenuState> = {
    no: { canUpdate: false, isNodeOpen: true, kind: 'node', nodeId: 'n1', x: 20, y: 20 },
    yes: { canUpdate: true, isNodeOpen: true, kind: 'node', nodeId: 'n1', x: 20, y: 20 },
  };
  const render = (canUpdate: boolean, onUpdate: () => void) => {
    const menuState = menuStates[canUpdate ? 'yes' : 'no'];

    return act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <NodeContextMenu
              canPaste={false}
              menuState={menuState}
              onAddConnector={vi.fn()}
              onClose={vi.fn()}
              onCopy={vi.fn()}
              onDelete={vi.fn()}
              onDuplicate={vi.fn()}
              onPaste={vi.fn()}
              onToggleOpen={vi.fn()}
              onUpdate={onUpdate}
            />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  };
  const updateItem = () => document.querySelector<HTMLElement>('[role="menuitem"][data-value="update"]');

  it('offers "Update node" only for a node whose template can be applied in place', async () => {
    const onUpdate = vi.fn();

    await render(false, onUpdate);
    expect(updateItem()).toBeNull();

    await render(true, onUpdate);
    expect(updateItem()?.textContent).toContain('nodes.updateNode');
    await act(() => updateItem()!.click());
    expect(onUpdate).toHaveBeenCalledOnce();
  });

  it('hints each selection action with the binding of the editor command it mirrors', async () => {
    await render(false, vi.fn());

    const hint = (value: string) =>
      document.querySelector(`[role="menuitem"][data-value="${value}"] kbd`)?.textContent ?? null;

    expect(hint('copy')).toBe('workflows.copySelection');
    expect(hint('paste')).toBe('workflows.pasteSelection');
    expect(hint('delete')).toBe('workflows.deleteSelection');
    expect(hint('duplicate')).toBe('workflows.duplicateSelection');
    expect(hint('toggle-open')).toBeNull();
  });

  it('offers both pastes on empty canvas, disabled with an empty clipboard, anchored where the menu opened', async () => {
    const onPaste = vi.fn();
    const renderPane = (canPaste: boolean) =>
      act(() =>
        root.render(
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <NodeContextMenu
                canPaste={canPaste}
                menuState={paneMenuState}
                onAddConnector={vi.fn()}
                onClose={vi.fn()}
                onCopy={vi.fn()}
                onDelete={vi.fn()}
                onDuplicate={vi.fn()}
                onPaste={onPaste}
                onToggleOpen={vi.fn()}
                onUpdate={vi.fn()}
              />
            </WorkflowUiProvider>
          </ChakraProvider>
        )
      );
    const item = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`)!;

    await renderPane(false);
    expect(item('paste').getAttribute('aria-disabled')).toBe('true');
    expect(item('paste-with-connections').getAttribute('aria-disabled')).toBe('true');

    await renderPane(true);
    await act(() => item('paste').click());
    await renderPane(true);
    await act(() => item('paste-with-connections').click());

    expect(onPaste.mock.calls).toEqual([
      [false, paneMenuState.position],
      [true, paneMenuState.position],
    ]);
  });
});

const paneMenuState = { kind: 'pane', position: { x: 140, y: -60 }, x: 20, y: 20 } as const;
