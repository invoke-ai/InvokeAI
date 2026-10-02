import { ChakraProvider } from '@chakra-ui/react';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { settleAnimations } from '@platform/browser/settleAnimations.testing';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it } from 'vitest';
import { page } from 'vitest/browser';

import { ProjectFileOptionsProvider, useProjectFileOptions } from './ProjectFileOptionsProvider';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let root: Root | undefined;
let host: HTMLDivElement | undefined;
let results: ({ includeFonts: boolean } | null)[] = [];
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Harness = () => {
  const { requestExportOptions } = useProjectFileOptions();
  return (
    <button
      type="button"
      onClick={() => {
        void requestExportOptions('Typography', captureAccountScope()).then((result) => results.push(result));
      }}
    >
      Export
    </button>
  );
};

const render = async () => {
  await page.viewport(1280, 900);
  results = [];
  accountLifecycle.activate('fonts-test');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <ProjectFileOptionsProvider>
            <Harness />
          </ProjectFileOptionsProvider>
        </I18nextProvider>
      </ChakraProvider>
    )
  );
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  accountLifecycle.invalidate();
});

it('defaults to references, includes files only when selected, and restores focus', async () => {
  await render();
  const trigger = page.getByRole('button', { name: 'Export', exact: true });
  await act(() => trigger.click());
  const checkbox = page.getByRole('checkbox', { name: 'Include font files' });
  await expect.element(checkbox).not.toBeChecked();
  await act(() => page.getByRole('button', { name: 'Export project', exact: true }).click());
  expect(results).toEqual([{ includeFonts: false }]);
  await expect.element(trigger).toHaveFocus();

  await act(() => trigger.click());
  await act(() => page.getByText('Include font files', { exact: true }).click());
  await expect.element(page.getByText(/permission to redistribute/)).toBeVisible();
  await settleAnimations();
  await page.screenshot({ path: '../../../../artifacts/fonts/export-dialog.png' });
  await act(() => page.getByRole('button', { name: 'Export project', exact: true }).click());
  expect(results).toEqual([{ includeFonts: false }, { includeFonts: true }]);
});

it('cancels a pending choice when its account goes away', async () => {
  await render();
  await act(() => page.getByRole('button', { name: 'Export', exact: true }).click());
  await expect.element(page.getByRole('dialog')).toBeVisible();
  await act(() => accountLifecycle.activate('another-account'));
  expect(results).toEqual([null]);
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
});
