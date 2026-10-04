import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { system } from '@theme/system';
import { act, useCallback, useMemo } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('./useProjectCardActions', () => ({
  useProjectCardActions: () => ({
    delete: () => Promise.resolve(),
    duplicate: () => {},
    export: () => {},
    rename: () => Promise.resolve(),
  }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const navigate = vi.hoisted(() => vi.fn());

vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => navigate,
  Link: ({ children, ...props }: { children?: unknown } & Record<string, unknown>) => (
    <a href="/app" {...props}>
      {children as never}
    </a>
  ),
}));

const { ProjectActionsMenuProvider, useProjectActionsMenu, useProjectActionsMenuTrigger } =
  await import('./ProjectActionsMenuHost');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const makeSummary = (id: string): ProjectSummary =>
  ({
    coverUrl: undefined,
    id,
    name: `Project ${id}`,
    minimumCanvasSchemaVersion: 3,
    schemaVersion: 1,
    updatedAt: 0,
  }) as unknown as ProjectSummary;

const NOOP_TOGGLE_PIN = () => {};

/** The card shape the launchpad project cards share: right-click anchor plus a dots trigger. */
const Card = ({ id }: { id: string }) => {
  const menu = useProjectActionsMenu();
  const summary = useMemo(() => makeSummary(id), [id]);
  const menuTarget = useMemo(() => ({ isPinned: false, onTogglePin: NOOP_TOGGLE_PIN, summary }), [summary]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => menu.openAtPointer(event, menuTarget),
    [menu, menuTarget]
  );
  const menuTrigger = useProjectActionsMenuTrigger(menuTarget);

  return (
    <div data-card={id} style={{ height: 120, position: 'relative', width: 200 }} onContextMenu={handleContextMenu}>
      <IconButton
        aria-expanded={menuTrigger.isExpanded}
        aria-haspopup="menu"
        aria-label={`actions ${id}`}
        size="sm"
        variant="ghost"
        onClick={menuTrigger.onClick}
        onPointerDown={menuTrigger.onPointerDown}
      >
        …
      </IconButton>
    </div>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const rightClick = async (element: Element) => {
  await act(async () => {
    const rect = element.getBoundingClientRect();
    const init = {
      bubbles: true,
      button: 2,
      cancelable: true,
      clientX: rect.left + 20,
      clientY: rect.top + 20,
    };
    element.dispatchEvent(new PointerEvent('pointerdown', init));
    element.dispatchEvent(new MouseEvent('contextmenu', init));
    element.dispatchEvent(new PointerEvent('pointerup', init));
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });
  });
};

const leftClick = async (element: Element) => {
  await act(async () => {
    const init = { bubbles: true, button: 0, cancelable: true };
    element.dispatchEvent(new PointerEvent('pointerdown', init));
    element.dispatchEvent(new PointerEvent('pointerup', init));
    element.dispatchEvent(new MouseEvent('click', init));
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });
  });
};

const openMenus = () => document.querySelectorAll('[role="menu"][data-state="open"]');
const dotsButton = (id: string) => document.querySelector(`button[aria-label="actions ${id}"]`)!;

describe('ProjectActionsMenuHost', () => {
  it('opens rename from a real pointer selection', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <ProjectActionsMenuProvider>
            <Card id="one" />
          </ProjectActionsMenuProvider>
        </ChakraProvider>
      )
    );
    await act(() => userEvent.click(dotsButton('one')));
    const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="rename"]')!;
    await act(() => userEvent.click(item));
    await vi.waitFor(() => {
      expect(document.querySelector('[role="dialog"][data-state="open"]')).not.toBeNull();
      expect(document.querySelector<HTMLInputElement>('input[name="renameValue"]')?.value).toBe('Project one');
    });
  });

  it('opens the Preferences intermediates section focused on the project', async () => {
    const { peekIntermediatesFocus } = await import('@features/intermediates/data/focus');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <ProjectActionsMenuProvider>
            <Card id="one" />
          </ProjectActionsMenuProvider>
        </ChakraProvider>
      )
    );
    await act(() => userEvent.click(dotsButton('one')));
    const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="intermediates"]')!;
    await act(() => userEvent.click(item));

    expect(peekIntermediatesFocus()).toEqual({ projectId: 'one' });
    // Revealing the manager's entry puts focus in it on the page.
    expect(navigate).toHaveBeenCalledWith({
      params: { section: 'intermediates' },
      search: { setting: 'intermediatesManager' },
      to: '/preferences/$section',
    });
    await vi.waitFor(() => expect(openMenus()).toHaveLength(0));
  });

  it('moves the one menu across cards instead of racing sibling layers', async () => {
    // A single host prevents Zag from treating the next card's menu as nested under the still-closing previous
    // menu.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <ProjectActionsMenuProvider>
            <Card id="one" />
            <Card id="two" />
          </ProjectActionsMenuProvider>
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const cardOne = host.querySelector('[data-card="one"]')!;
    const cardTwo = host.querySelector('[data-card="two"]')!;

    await rightClick(cardOne);
    expect(openMenus()).toHaveLength(1);
    expect(dotsButton('one').getAttribute('aria-expanded')).toBe('true');

    await rightClick(cardTwo);
    expect(openMenus()).toHaveLength(1);
    expect(dotsButton('one').getAttribute('aria-expanded')).toBe('false');
    expect(dotsButton('two').getAttribute('aria-expanded')).toBe('true');

    // Re-right-clicking the open card's own body keeps its menu open too.
    await rightClick(cardTwo);
    expect(openMenus()).toHaveLength(1);
    expect(dotsButton('two').getAttribute('aria-expanded')).toBe('true');

    // A second dots click closes; pointerdown dismissal must not cause click to reopen.
    await leftClick(dotsButton('one'));
    expect(openMenus()).toHaveLength(1);
    expect(dotsButton('one').getAttribute('aria-expanded')).toBe('true');

    await leftClick(dotsButton('one'));
    expect(openMenus()).toHaveLength(0);
    expect(dotsButton('one').getAttribute('aria-expanded')).toBe('false');
  });
});
