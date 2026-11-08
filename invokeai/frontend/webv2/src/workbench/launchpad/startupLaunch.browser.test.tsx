/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { beginStartupLaunch, openStartupLaunch, settleStartupLaunch } from './startupLaunch';
import { StartupLaunchSettled } from './StartupLaunchSettled';
import { StayOnLaunchpadButton } from './StayOnLaunchpadButton';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

/** A page load that marked a launch into the editor and has not rendered it yet. */
const launchIntoEditor = () => {
  expect(beginStartupLaunch()).toBe(true);
  openStartupLaunch('project-1');
};
/** What the next launch decides: false when the mark made it stay home. */
const nextLaunchProceeds = (): boolean => {
  const proceeds = beginStartupLaunch();
  settleStartupLaunch();
  return proceeds;
};

const render = async (content: React.ReactNode, path = '/app') => {
  const rootRoute = createRootRoute({ component: () => <Outlet /> });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
    routeTree: rootRoute.addChildren([
      createRoute({ getParentRoute: () => rootRoute, path: '/' }),
      createRoute({ component: () => content, getParentRoute: () => rootRoute, path: 'app' }),
    ]),
  });
  await act(async () => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <RouterProvider router={router} />
        </I18nextProvider>
      </ChakraProvider>
    );
    await router.load();
  });
  return router;
};

beforeEach(() => {
  window.localStorage.clear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  settleStartupLaunch();
  window.localStorage.clear();
});

describe('startup launch in the editor', () => {
  it('lets the next launch reopen the work once the editor has rendered', async () => {
    launchIntoEditor();
    await render(<StartupLaunchSettled />);

    expect(nextLaunchProceeds()).toBe(true);
  });

  it('offers the Launchpad only while a launch opens, and choosing it is not a failed open', async () => {
    const router = await render(<StayOnLaunchpadButton />);
    const stay = page.getByRole('button', { exact: true, name: i18n.t('splash.stayOnLaunchpad') });
    expect(stay.query()).toBeNull();

    await act(() => launchIntoEditor());
    await userEvent.click(stay);

    await expect.poll(() => router.state.location.pathname).toBe('/');
    expect(nextLaunchProceeds()).toBe(true);
  });
});
