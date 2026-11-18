/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';
import type * as workbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { galleryWidgetManifest } from './manifest';
import { Field } from './settingsBindings';

/** The workbench the binding sees; null is the Launchpad, where there is none. */
const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));
const subscribeToNothing = () => () => {};
const getNull = () => null;
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useOptionalWorkbenchCommands: () => storeRef.current?.commands ?? null,
  useOptionalWorkbenchQueries: () => storeRef.current?.queries ?? null,
  useOptionalWorkbenchSelector: (selector: (snapshot: WorkbenchSnapshot) => unknown, fallback: unknown) => {
    const store = storeRef.current;
    return useExternalStoreSelector<WorkbenchSnapshot | null, unknown>(
      store?.subscribe ?? subscribeToNothing,
      store?.getSnapshot ?? getNull,
      (snapshot) => (snapshot ? selector(snapshot) : fallback)
    );
  },
}));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: translations } } });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
const field = (id: string) => galleryWidgetManifest.settings!.fields.find((entry) => entry.id === id)!;
const renderBoth = (id: string, target?: SettingsTarget) =>
  act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <div data-surface="quick">
            <Field field={field(id)} surface="quick" target={target} />
          </div>
          <div data-surface="dialog">
            <Field field={field(id)} surface="dialog" target={target} />
          </div>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
const surface = (name: 'quick' | 'dialog', selector: string) =>
  host.querySelector<HTMLElement>(`[data-surface="${name}"] ${selector}`)!;
const galleryPins = () => storeRef.current!.getSnapshot().activeProject.widgetInstances.gallery!.state.values;
const mountProject = (values: Record<string, unknown> = {}) => {
  const store = createWorkbenchStore();
  storeRef.current = store;
  store.commands.widgets.patchInstanceValues('gallery', values);
  return { projectId: store.getSnapshot().activeProject.id };
};

beforeEach(async () => {
  accountLifecycle.activate('gallery-settings-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  storeRef.current = null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('Gallery settings in a project', () => {
  it('shows 100 minus the stored density and pins the inverse across both presentations', async () => {
    const target = mountProject({ imageDensityPercent: 25 });
    await renderBoth('imageSize', target);
    expect(surface('quick', '[role="slider"]').getAttribute('aria-valuenow')).toBe('75');
    expect(surface('dialog', '[role="slider"]').getAttribute('aria-valuenow')).toBe('75');

    await act(() => surface('quick', '[role="slider"]').focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(galleryPins().imageDensityPercent).toBe(24);
    expect(surface('dialog', '[role="slider"]').getAttribute('aria-valuenow')).toBe('76');
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});
  });

  it("reads an earlier build's stored follow as off and pins the toggle", async () => {
    const target = mountProject({ autoAddBoardId: 'follow', selectedBoardId: 'picked' });
    await renderBoth('followSelectedBoard', target);
    const toggle = surface('quick', 'input') as HTMLInputElement;
    expect(toggle.checked).toBe(false);
    await act(() => toggle.click());
    expect(galleryPins().followSelectedBoard).toBe(true);
  });

  it('returns a paginated gallery to its first page when the order changes or follows the default', async () => {
    const target = mountProject({ galleryPage: 3, paginationMode: 'paginated' });
    await renderBoth('imageOrderDir', target);
    await act(() => surface('quick', '[role="combobox"]').click());
    await act(() => surface('quick', '[role="option"][data-value="ASC"]').click());
    expect(galleryPins()).toMatchObject({ galleryPage: 0, imageOrderDir: 'ASC' });

    storeRef.current!.commands.gallery.setPage(2);
    await act(() =>
      page
        .getByRole('button', {
          name: i18n.t('settingsDialog.layer.pinnedNamed', { setting: i18n.t('widgets.gallery.imageSort') }),
        })
        .first()
        .click()
    );
    await act(() => page.getByRole('menuitem', { name: /Follow my default/ }).click());
    expect(Object.hasOwn(galleryPins(), 'imageOrderDir')).toBe(false);
    expect(galleryPins().galleryPage).toBe(0);
  });

  it('pins the project gallery from an alias instance, sharing its pins with every gallery cog', async () => {
    const { projectId } = mountProject({ showImageDimensions: false });
    const store = storeRef.current!;
    const project = () => store.getSnapshot().activeProject;
    const centerTarget = { instanceId: 'gallery:center', projectId };
    await renderBoth('showImageDimensions', centerTarget);
    const toggle = surface('quick', 'input') as HTMLInputElement;
    await act(() => toggle.click());
    expect(getProjectWidgetValues(project(), 'gallery').showImageDimensions).toBe(true);
    expect(Object.hasOwn(project().widgetInstances['gallery:center']!.state.values, 'showImageDimensions')).toBe(false);

    await renderBoth('showPendingItems', { projectId });
    await act(() => (surface('quick', 'input') as HTMLInputElement).click());
    await renderBoth('showPendingItems', centerTarget);
    expect(
      page
        .getByRole('button', {
          name: i18n.t('settingsDialog.layer.pinnedNamed', { setting: i18n.t('widgets.gallery.showPendingItems') }),
        })
        .elements()
    ).toHaveLength(2);
  });

  it('disables edits for a target belonging to another project', async () => {
    mountProject();
    await renderBoth('showPendingItems', { projectId: 'old-project' });
    const toggle = surface('quick', 'input') as HTMLInputElement;
    expect(toggle.disabled).toBe(true);
    await act(() => toggle.click());
    expect(Object.hasOwn(galleryPins(), 'showPendingItems')).toBe(false);
  });
});

describe('Gallery settings in the Launchpad', () => {
  it('edits the user default, storing the size as its density', async () => {
    await renderBoth('imageSize');
    expect(surface('quick', '[role="slider"]').getAttribute('aria-valuenow')).toBe('50');
    await act(() => surface('quick', '[role="slider"]').focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({ gallery: { imageDensityPercent: 49 } });
    expect(surface('dialog', '[role="slider"]').getAttribute('aria-valuenow')).toBe('51');
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});
  });
});
