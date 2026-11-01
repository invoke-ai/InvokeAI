/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { StoredAspectRatioPreset } from '@workbench/settings/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { AspectRatioPicker } from '@features/generation/components';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { useAspectRatioPresets } from './aspectRatioPresets';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The account's synced preferences live in the backend's client state; this keeps them in memory.
vi.mock('@workbench/projects/api', async (importOriginal) => {
  const clientState = new Map<string, string>();
  return {
    ...(await importOriginal<object>()),
    deleteClientStateValue: (key: string) => Promise.resolve(void clientState.delete(key)),
    getClientStateValue: (key: string) => Promise.resolve(clientState.get(key) ?? null),
    setClientStateValue: (key: string, value: string) => Promise.resolve(void clientState.set(key, value)),
  };
});

let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  // The global instance: the Undo toast reads its button label from it.
  await i18next.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

beforeEach(() => {
  accountLifecycle.activate('aspect-ratio-presets');
});

afterEach(async () => {
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  // Clears the account's settings with the rest of its resources.
  accountLifecycle.invalidate();
});

const preset = (id: string, label: string, width: number, height: number): StoredAspectRatioPreset => ({
  height,
  id,
  label,
  width,
});

const Picker = () => (
  <AspectRatioPicker
    currentSize={{ height: 640, width: 1536 }}
    getSizeForRatio={() => ({ height: 640, width: 1536 })}
    lockedRatio={null}
    presets={useAspectRatioPresets()}
    onChange={() => undefined}
  />
);

const render = async (stored: StoredAspectRatioPreset[]) => {
  await patchWorkbenchPreferences({ aspectRatioPresets: stored });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18next}>
        <ChakraProvider value={system}>
          <Picker />
          <AppToaster />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  await userEvent.click(page.getByRole('button', { name: /^Aspect ratio/ }));
};

/** The saved rows' names, in the order the picker lists them. */
const rowNames = () =>
  [...document.querySelectorAll<HTMLButtonElement>('button[aria-label$="1536×640"]')].map(
    (row) => row.getAttribute('aria-label')?.split(',')[0]
  );

describe('useAspectRatioPresets', () => {
  it('deletes a saved ratio with an Undo that puts it back where it was', async () => {
    const stored = [preset('a', 'Banner', 12, 5), preset('b', 'Cinema', 2.39, 1), preset('c', 'Tall', 2, 3)];
    await render(stored);

    await userEvent.click(page.getByRole('button', { name: 'Delete ratio Cinema (2.39:1)' }));

    await expect.poll(rowNames).toEqual(['Banner', 'Tall']);
    expect(getWorkbenchPreferences().aspectRatioPresets).toEqual([stored[0], stored[2]]);

    await expect.element(page.getByText('Deleted ratio Cinema')).toBeVisible();
    await userEvent.click(page.getByRole('button', { name: 'Undo' }));

    await expect.poll(() => getWorkbenchPreferences().aspectRatioPresets).toEqual(stored);
  });

  it('adds and edits through the account preference, keeping entries only a newer build can use', async () => {
    // 20:1 is past this build's 8:1, so it is not listed, but a newer build's ratio must survive this build's edits.
    const wide = preset('wide', 'Panorama', 20, 1);
    await render([preset('a', 'Banner', 12, 5), wide]);

    expect(rowNames()).toEqual(['Banner']);

    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));
    await userEvent.keyboard('Same{Enter}');
    await userEvent.click(page.getByRole('button', { name: 'Edit ratio Banner (12:5)' }));
    await userEvent.clear(page.getByRole('textbox', { name: 'Ratio name' }));
    await userEvent.keyboard('Header{Enter}');

    await expect.poll(rowNames).toEqual(['Header', 'Same']);
    const [edited, kept, added] = getWorkbenchPreferences().aspectRatioPresets;
    expect(edited).toEqual(preset('a', 'Header', 12, 5));
    expect(kept).toEqual(wide);
    expect(added).toMatchObject({ height: 5, label: 'Same', width: 12 });
  });

  it('offers no Add ratio once the list is full', async () => {
    await render(Array.from({ length: 50 }, (_, index) => preset(`p${index}`, `Ratio ${index}`, 2, 1)));

    await expect.element(page.getByText('You can keep up to 50 ratios. Delete one to add another.')).toBeVisible();
    expect(page.getByRole('button', { name: 'Add ratio' }).elements()).toHaveLength(0);
  });
});
