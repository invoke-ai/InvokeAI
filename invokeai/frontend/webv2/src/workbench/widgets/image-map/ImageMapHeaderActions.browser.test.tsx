import type { WidgetInstanceRuntimeMeta, WidgetRuntimeApi } from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-jsx-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { WidgetHeaderActionsGroup } from '@workbench/widget-frame/WidgetFrames';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const workbench = vi.hoisted(() => ({
  galleryValues: {} as Record<string, unknown>,
  updateSettings: vi.fn(),
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useActiveProjectId: () => 'project-1',
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ id: 'project-1', widgetInstances: {}, widgetRegions: {} }),
  useWidgetValuesSelector: (widgetId: string, select: (values: Record<string, unknown>) => unknown) =>
    select(widgetId === 'gallery' ? workbench.galleryValues : {}),
  useWorkbenchCommands: () => ({
    gallery: { updateSettings: workbench.updateSettings },
    widgets: { float: vi.fn(), patchValues: vi.fn() },
  }),
  useWorkbenchQueries: () => ({ getProject: () => null, isActiveProject: () => true }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ i18n: { language: 'en' }, t: (key: string) => key }),
}));

import { ImageMapHeaderActions } from './ImageMapHeaderActions';
import { imageMapWidgetManifest } from './manifest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const instance = { id: 'image-map-1', typeId: 'image-map' } as unknown as WidgetInstanceRuntimeMeta;
const runtime = {} as WidgetRuntimeApi;

const renderHeader = async () => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <WidgetHeaderActionsGroup
          actions={
            <ImageMapHeaderActions
              instance={instance}
              manifest={imageMapWidgetManifest}
              region="right"
              runtime={runtime}
            />
          }
          instance={instance}
          manifest={imageMapWidgetManifest}
          region="right"
          runtime={runtime}
        />
      </ChakraProvider>
    )
  );
};

beforeEach(() => {
  workbench.galleryValues = {};
  workbench.updateSettings.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('image map header actions', () => {
  it('puts the shared media filter directly left of the settings button and writes the Gallery setting', async () => {
    workbench.galleryValues = { mediaFilter: 'image' };
    await renderHeader();

    const buttons = [...(host?.querySelectorAll<HTMLButtonElement>('button') ?? [])];
    const filterIndex = buttons.findIndex((button) => button.hasAttribute('data-gallery-media-filter-menu'));
    const settingsIndex = buttons.findIndex((button) => button.getAttribute('aria-label') === 'widgets.settingsLabel');

    expect(filterIndex, 'media filter did not render in the header').toBeGreaterThanOrEqual(0);
    expect(buttons[filterIndex]?.getAttribute('data-media-filter')).toBe('image');
    // The view's "Show all media" hands focus to this id.
    expect(buttons[filterIndex]?.id).toBe('image-map-media-filter-trigger');
    expect(settingsIndex, 'settings button did not render').toBe(filterIndex + 1);

    buttons[filterIndex]?.focus();
    await userEvent.keyboard('{Enter}');
    await page.getByRole('menuitemradio', { name: 'widgets.gallery.mediaFilterVideos' }).click();

    expect(workbench.updateSettings).toHaveBeenCalledExactlyOnceWith({ mediaFilter: 'video' });
  });
});
