import type { GalleryBoard } from '@features/gallery/core/types';
import type { GalleryUiAdapter } from '@features/gallery/react';
import type { WidgetInstanceRuntimeMeta, WidgetRuntimeApi } from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-jsx-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { getGallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { GalleryUiProvider } from '@features/gallery/react';
import { GalleryWidgetHeaderActions, GalleryWidgetLabel } from '@features/gallery/widget';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { galleryWidgetManifest } from '@workbench/widgets/gallery/manifest';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useActiveProjectId: () => 'project-1',
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({
      id: 'project-1',
      invocation: { sourceId: null, sourceLocked: false },
      widgetInstances: {},
      widgetRegions: {},
    }),
  useWorkbenchCommands: () => ({ widgets: { float: vi.fn(), patchValues: vi.fn() } }),
  useWorkbenchQueries: () => ({ getProject: () => null, isActiveProject: () => true }),
}));

import { WidgetHeader } from './WidgetFrames';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const board = (id: string, name: string, overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  access: { actions: ['read', 'add_media'], adminOverride: false, generalAccess: 'restricted', role: 'owner' },
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  name,
  projectId: 'project-1',
  videoCount: 0,
  ...overrides,
});

const instance = { id: 'gallery', typeId: 'gallery' } as unknown as WidgetInstanceRuntimeMeta;
const runtime = {} as WidgetRuntimeApi;
let host: HTMLDivElement | null = null;
let root: Root | null = null;

const renderGalleryHeader = async (width: number, boardName: string, destinationName: string) => {
  const queryClient = new QueryClient();
  const values = { autoAddBoardId: 'destination', selectedBoardId: 'browsed' };
  const settings = getGallerySettings(values);
  queryClient.setQueryData(
    galleryBoardsOptions({
      includeArchived: settings.showArchivedBoards,
      includeDateBoards: settings.showDateBoards,
      orderBy: settings.boardOrderBy,
      orderDir: settings.boardOrderDir,
    }).queryKey,
    [board('browsed', boardName), board('destination', destinationName)]
  );
  const adapter = {
    gallery: { updateSettings: vi.fn() },
    galleryValues: values,
    projectId: 'project-1',
    projectName: 'Project',
    projects: [],
  } as unknown as GalleryUiAdapter;
  host!.style.width = `${width}px`;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={queryClient}>
            <GalleryUiProvider adapter={adapter}>
              <WidgetHeader
                HeaderLabel={GalleryWidgetLabel}
                actions={<GalleryWidgetHeaderActions />}
                instance={instance}
                manifest={galleryWidgetManifest}
                region="right"
                runtime={runtime}
              />
            </GalleryUiProvider>
          </QueryClientProvider>
        </I18nextProvider>
      </ChakraProvider>
    )
  );
};

beforeEach(() => {
  accountLifecycle.activate('widget-header-layout-test');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
});

const intersects = (left: DOMRect, right: DOMRect) =>
  left.left < right.right && right.left < left.right && left.top < right.bottom && right.top < left.bottom;

describe('gallery widget header', () => {
  // 350px is the narrowest a side rail gets; 600px leaves room for every part.
  it.each([350, 420, 600])(
    'keeps the board and the destination apart at %ipx, truncating their names',
    async (width) => {
      await renderGalleryHeader(width, 'Mahogany House concept explorations', 'Client review selects for April');

      const header = host!.firstElementChild!.getBoundingClientRect();
      const boardButton = host!.querySelector<HTMLElement>('button[aria-label^="Toggle"]')!;
      const destination = host!.querySelector<HTMLElement>('[data-gallery-destination]')!;
      const destinationRect = destination.getBoundingClientRect();
      const settings = host!.querySelector<HTMLElement>('button[aria-label="Gallery settings"]')!;

      // Every part of the label, the board's cover and chevron included, ends before Saving to starts.
      for (const part of [boardButton, ...boardButton.querySelectorAll('*')]) {
        expect(intersects(part.getBoundingClientRect(), destinationRect), part.outerHTML.slice(0, 80)).toBe(false);
      }
      expect(destinationRect.right).toBeLessThanOrEqual(settings.getBoundingClientRect().left);
      expect(settings.getBoundingClientRect().right).toBeLessThanOrEqual(header.right);
      // Neither long name fits: both truncate inside their buttons.
      for (const [button, name] of [
        [boardButton, 'Mahogany House concept explorations'],
        [destination, 'Client review selects for April'],
      ] as const) {
        const text = button.querySelector<HTMLElement>(`[title="${name}"]`)!;
        // The head gives way with an ellipsis; the tail stays whole.
        const head = text.firstElementChild!;
        expect(head.scrollWidth).toBeGreaterThan(head.clientWidth);
        expect(text.getBoundingClientRect().right).toBeLessThanOrEqual(button.getBoundingClientRect().right);
      }
    }
  );

  it('says "Saving to" only where the header has room for it', async () => {
    const prefix = () =>
      [...destinationTrigger().querySelectorAll('span')].find((span) => span.textContent === 'Saving to ');
    const destinationTrigger = () => host!.querySelector<HTMLElement>('[data-gallery-destination]')!;

    await renderGalleryHeader(600, 'Concepts', 'Selects');
    expect(prefix()!.getClientRects()).toHaveLength(1);

    await renderGalleryHeader(350, 'Concepts', 'Selects');
    expect(prefix()!.getClientRects()).toHaveLength(0);
    expect(destinationTrigger().getAttribute('aria-label')).toMatch(/^Saving to Selects\./);
  });
});
