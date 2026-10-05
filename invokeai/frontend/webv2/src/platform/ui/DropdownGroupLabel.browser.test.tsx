import { ChakraProvider, Menu } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('dropdown group labels', () => {
  it('keeps headings more compact than interactive items', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Menu.Root open>
            <Menu.Positioner>
              <Menu.Content>
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel data-testid="group-label">Group</Menu.ItemGroupLabel>
                  <Menu.Item data-testid="menu-item" value="item">
                    Item
                  </Menu.Item>
                </Menu.ItemGroup>
              </Menu.Content>
            </Menu.Positioner>
          </Menu.Root>
        </ChakraProvider>
      );
      await Promise.resolve();
    });

    const label = host.querySelector<HTMLElement>('[data-testid="group-label"]');
    const item = host.querySelector<HTMLElement>('[data-testid="menu-item"]');

    expect(label).not.toBeNull();
    expect(item).not.toBeNull();
    expect(label!.getBoundingClientRect().height).toBeLessThan(item!.getBoundingClientRect().height);
  });
});
