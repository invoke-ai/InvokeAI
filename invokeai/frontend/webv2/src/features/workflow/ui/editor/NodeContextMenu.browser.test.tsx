import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { NodeContextMenu, type NodeContextMenuState } from './NodeContextMenu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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
});
