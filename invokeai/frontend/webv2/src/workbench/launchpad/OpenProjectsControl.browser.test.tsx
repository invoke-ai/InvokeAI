/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as openProjects from '@workbench/projects/openProjects';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const snapshot = vi.hoisted(() => ({
  current: { activeId: 'three', ids: ['one', 'two', 'three'], status: 'ready' } as openProjects.OpenProjectsSnapshot,
}));

vi.mock('@workbench/projects/openProjects', () => ({
  refreshOpenProjects: () => Promise.resolve(),
  useOpenProjectsSelector: <T,>(select: (value: openProjects.OpenProjectsSnapshot) => T) => select(snapshot.current),
}));

vi.mock('@workbench/projects/library', () => ({
  useProjectLibrarySelector: <T,>(select: (value: { summaries: { id: string; name: string }[] }) => T) =>
    select({
      summaries: [
        { id: 'one', name: 'First' },
        { id: 'two', name: 'Second' },
        { id: 'three', name: 'Third' },
      ],
    }),
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

vi.mock('@tanstack/react-router', () => ({
  Link: ({ children, search, to, ...props }: { children?: unknown; search?: { project?: string }; to: string }) => (
    <a data-project={search?.project} href={to} {...props}>
      {children as never}
    </a>
  ),
}));

const { OpenProjectsNavSection } = await import('./OpenProjectsControl');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const renderedOrder = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <OpenProjectsNavSection headingCss={{}} itemProps={{ justifyContent: 'start', size: 'sm', w: 'full' }} />
      </ChakraProvider>
    )
  );
  return [...host.querySelectorAll('a[data-project]')].map((link) => link.getAttribute('data-project'));
};

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('Launchpad open projects', () => {
  it('lists the current project first and keeps the others in their open order', async () => {
    snapshot.current = { activeId: 'three', ids: ['one', 'two', 'three'], status: 'ready' };
    expect(await renderedOrder()).toEqual(['three', 'one', 'two']);
  });

  it('keeps the open order when no open project is current', async () => {
    snapshot.current = { activeId: 'gone', ids: ['one', 'two', 'three'], status: 'ready' };
    expect(await renderedOrder()).toEqual(['one', 'two', 'three']);
  });
});
