import type { GalleryMediaUsageReport } from '@features/gallery/core/mediaUsage';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard, GalleryBoardAccess, GalleryBoardAction } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardMenu } from './GalleryBoardMenu';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const messages: Record<string, string> = {
        'widgets.gallery.assetCount': `${String(values?.count)} uploads`,
        'common.cancel': 'Cancel',
        'common.retry': 'Retry',
        'widgets.gallery.deleteBoard': 'Delete board',
        'widgets.gallery.deleteBoardAndMedia': `Delete board and ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardDescription': `Its ${String(values?.media)} move to Uncategorized.`,
        'widgets.gallery.deleteBoardEmptyDescription': 'The board is empty.',
        'widgets.gallery.deleteBoardWithMediaDescription': `Its ${String(values?.media)} are deleted with it.`,
        'widgets.gallery.deleteBoardWithOwnMediaDescription': `Your ${String(
          values?.media
        )} on it are deleted with it; the rest move to Uncategorized.`,
        'widgets.gallery.deleteBoardWithNoOwnMediaDescription':
          'None of its media is yours; all of it moves to Uncategorized.',
        'widgets.gallery.deleteBoardIncludeOwnMedia': `Also permanently delete your ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardIncludeOwnMediaNone': 'Also permanently delete your items',
        'widgets.gallery.deleteBoardAndOwnMedia': `Delete board and your ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardIncludeMedia': `Also permanently delete its ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardQuestion': `Delete board "${String(values?.name)}"?`,
        'widgets.gallery.deleteBoardTypeName': `Type "${String(values?.name)}" to confirm`,
        'widgets.gallery.deleteInUse.checking': 'Checking where this media is used…',
        'widgets.gallery.deleteInUse.none': 'Nothing uses this media.',
        'widgets.gallery.deleteInUse.failed': 'Couldn’t check where this media is used.',
        'widgets.gallery.deleteInUse.group.project': `Project “${String(values?.name)}”`,
        'widgets.gallery.deleteInUse.itemCount': `${String(values?.count)} items`,
        'widgets.gallery.deleteInUse.summaryBoard': `${String(values?.count)} of the ${String(
          values?.media
        )} this deletes are still in use:`,
        'widgets.gallery.deleteInUse.summaryOne': 'Still in use:',
        'widgets.gallery.deleteInUse.surface.canvasRaster': 'Canvas raster layers',
        'widgets.gallery.downloadBoard': 'Download board',
        'widgets.gallery.downloadBoardWithOmission': `Download board (${String(values?.count)} video omitted)`,
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
        'widgets.gallery.mediaCountsThree': `${String(values?.first)}, ${String(values?.second)} and ${String(
          values?.third
        )}`,
        'widgets.gallery.mediaCountsTwo': `${String(values?.first)} and ${String(values?.second)}`,
        'widgets.gallery.videoCount': `${String(values?.count)} videos`,
        'widgets.gallery.boardGroups.library': 'Library',
        'widgets.gallery.boardGroups.projects': 'Projects',
        'widgets.gallery.moveBoard': 'Move to…',
      };

      return messages[key] ?? key;
    },
  }),
}));

const board = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'board-1',
  imageCount: 2,
  kind: 'board',
  name: 'Board 1',
  isInbox: false,
  projectId: null,
  videoCount: 1,
} as const;
const target = { board, x: 20, y: 20 };
const noop = vi.fn();
const context = {
  actions: {
    archiveBoard: vi.fn(),
    deleteBoard: vi.fn(),
    downloadBoard: vi.fn(),
    getBoardMediaUsage: vi.fn<(boardId: string, signal: AbortSignal) => Promise<GalleryMediaUsageReport>>(),
    moveBoard: vi.fn(),
    renameBoard: vi.fn(),
    setBoardGeneralAccess: vi.fn(),
    updateSettings: vi.fn(),
  },
  gallery: {
    boards: [board],
    projectBoardId: null,
    selectedBoardId: 'board-1',
    settings: { autoAddBoardId: null, followSelectedBoard: false },
  },
  projectId: 'p1',
  projectName: 'Mahogany House',
  projectNames: new Map([
    ['p1', 'Mahogany House'],
    ['p2', 'Harbor Tower'],
  ]),
} as unknown as GalleryWidgetContextValue;
const targetContext = {
  ...context,
  gallery: { ...context.gallery, settings: { autoAddBoardId: 'board-1', followSelectedBoard: false } },
} as unknown as GalleryWidgetContextValue;

let host: HTMLDivElement | null = null;
let queryClient = new QueryClient();
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  queryClient = new QueryClient();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <GalleryWidgetContext value={context}>
            <GalleryBoardMenu target={target} onClose={noop} />
          </GalleryWidgetContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('states the exact number of videos omitted from the image-only board archive', () => {
  expect(document.body.textContent).toContain('Download board (1 video omitted)');
});

it('offers a plain board download when the board holds no videos', async () => {
  await renderMenuFor({ ...board, videoCount: 0 } as GalleryBoard);

  expect(menuItemLabels()).toContain('Download board');
  expect(document.body.textContent).not.toContain('omitted');
});

const openDeleteDialog = async () => {
  const deleteItem = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Delete board'
  );

  expect(deleteItem).toBeDefined();
  await act(() => userEvent.click(deleteItem!));
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).not.toBeNull());

  return document.querySelector<HTMLElement>('[role="alertdialog"]')!;
};

const dialogButton = (label: string) =>
  Array.from(document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')).find(
    (button) => button.textContent === label
  );

it('deletes the board and keeps its media unless the user also asks to delete the media', async () => {
  const dialog = await openDeleteDialog();

  // Only the kinds the board holds, and where they go.
  expect(dialog.textContent).toContain('Its 2 images and 1 videos move to Uncategorized.');
  expect(dialog.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(false);
  await act(() => userEvent.click(dialogButton('Delete board')!));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', false);
});

it('deletes the media too only behind its checkbox, naming how many items go', async () => {
  const dialog = await openDeleteDialog();

  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  expect(dialog.textContent).toContain('Also permanently delete its 3 items');
  // The description follows the choice: the media no longer moves anywhere.
  expect(dialog.textContent).toContain('Its 2 images and 1 videos are deleted with it.');
  expect(dialog.textContent).not.toContain('move to Uncategorized');
  await act(() => userEvent.click(dialogButton('Delete board and 3 items')!));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

it('asks for the board name before deleting many items with it', async () => {
  await renderMenuFor({ ...board, assetCount: 40, imageCount: 20 } as GalleryBoard);
  const dialog = await openDeleteDialog();

  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  const confirm = dialogButton('Delete board and 61 items')!;
  expect(confirm.disabled).toBe(true);

  await act(() => userEvent.type(dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])')!, 'Board 1'));
  expect(confirm.disabled).toBe(false);
  await act(() => userEvent.click(confirm));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

it('starts each opening of the delete dialog over, for the same board too', async () => {
  await renderMenuFor({ ...board, assetCount: 40, imageCount: 20 } as GalleryBoard);
  let dialog = await openDeleteDialog();

  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  await act(() => userEvent.type(dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])')!, 'Board 1'));
  expect(dialogButton('Delete board and 61 items')!.disabled).toBe(false);
  await act(() => userEvent.click(dialogButton('Cancel')!));
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());

  dialog = await openDeleteDialog();

  expect(dialog.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(false);
  expect(dialog.querySelector('input:not([type="checkbox"])')).toBeNull();
  expect(dialogButton('Delete board and 61 items')).toBeUndefined();
  expect(dialogButton('Delete board')!.disabled).toBe(false);

  // Asking again for the media brings back an empty name field and a refused delete.
  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  expect(dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])')!.value).toBe('');
  expect(dialogButton('Delete board and 61 items')!.disabled).toBe(true);
});

const boardUsageReport: GalleryMediaUsageReport = {
  checkedCount: 3,
  groups: [{ count: 2, kind: 'project', name: 'Harbor Tower', surfaces: [{ count: 2, surface: 'canvasRaster' }] }],
  inUseCount: 2,
  onlyImages: false,
  scope: 'board',
  serverCheck: 'complete',
};

/** The report's code loads lazily on the first settled check; a cold transform under a busy run can take a while. */
const REPORT_WAIT = { timeout: 5000 };

const tickDeleteMedia = (dialog: HTMLElement) =>
  act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));

it('checks where the media is used once deleting it is asked for, and shows the report beside the choice', async () => {
  const getBoardMediaUsage = vi.mocked(context.actions.getBoardMediaUsage);
  let settle: (report: GalleryMediaUsageReport) => void = () => {};
  getBoardMediaUsage.mockReset().mockImplementation(
    () =>
      new Promise((resolve) => {
        settle = resolve;
      })
  );
  const dialog = await openDeleteDialog();

  // Keeping the media breaks nothing, so nothing is checked for it.
  expect(getBoardMediaUsage).not.toHaveBeenCalled();
  await tickDeleteMedia(dialog);

  expect(getBoardMediaUsage).toHaveBeenCalledExactlyOnceWith('board-1', expect.any(AbortSignal));
  expect(dialog.textContent).toContain('Checking where this media is used…');
  expect(dialogButton('Delete board and 3 items')!.disabled).toBe(false);

  await act(() => settle(boardUsageReport));

  await vi.waitFor(
    () => expect(dialog.textContent).toContain('2 of the 3 items this deletes are still in use:'),
    REPORT_WAIT
  );
  expect(dialog.textContent).toContain('Project “Harbor Tower”2 items');
  expect(dialog.textContent).toContain('Canvas raster layers2 items');
  await act(() => userEvent.click(dialogButton('Delete board and 3 items')!));
  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

it('warns when the check failed, still lets the board and its media go, and checks again on Retry', async () => {
  const getBoardMediaUsage = vi.mocked(context.actions.getBoardMediaUsage);
  let settleRetry: (report: GalleryMediaUsageReport) => void = () => {};
  getBoardMediaUsage
    .mockReset()
    .mockRejectedValueOnce(new Error('offline'))
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          settleRetry = resolve;
        })
    );
  const dialog = await openDeleteDialog();

  await tickDeleteMedia(dialog);

  await vi.waitFor(() => expect(dialog.textContent).toContain('Couldn’t check where this media is used.'), REPORT_WAIT);
  expect(dialogButton('Delete board and 3 items')!.disabled).toBe(false);

  await act(() => userEvent.click(dialogButton('Retry')!));

  // Retry gives way to the checking state; focus stays in the dialog, on the panel that reads the outcome.
  const panel = dialog.querySelector<HTMLElement>('[aria-live="polite"]');
  await vi.waitFor(() => expect(dialog.textContent).toContain('Checking where this media is used…'));
  expect(document.activeElement).toBe(panel);

  await act(() => settleRetry({ ...boardUsageReport, checkedCount: 1, inUseCount: 1, onlyImages: true }));

  await vi.waitFor(() => expect(dialog.textContent).toContain('Still in use:'), REPORT_WAIT);
  expect(document.activeElement).toBe(panel);
  expect(dialog.contains(document.activeElement)).toBe(true);
  expect(getBoardMediaUsage).toHaveBeenCalledTimes(2);
});

it('asks once per opening: not again when ticked twice, but again when the dialog reopens', async () => {
  const getBoardMediaUsage = vi.mocked(context.actions.getBoardMediaUsage);
  getBoardMediaUsage.mockReset().mockResolvedValue(boardUsageReport);
  let dialog = await openDeleteDialog();

  await tickDeleteMedia(dialog);
  await vi.waitFor(() => expect(dialog.textContent).toContain('still in use'), REPORT_WAIT);
  await tickDeleteMedia(dialog);
  expect(dialog.textContent).not.toContain('still in use');
  await tickDeleteMedia(dialog);
  expect(dialog.textContent).toContain('still in use');
  expect(getBoardMediaUsage).toHaveBeenCalledOnce();

  await act(() => userEvent.click(dialogButton('Cancel')!));
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
  dialog = await openDeleteDialog();
  await tickDeleteMedia(dialog);

  await vi.waitFor(() => expect(getBoardMediaUsage).toHaveBeenCalledTimes(2));
});

/** A check that never answers, recording the signal of each call. */
const hangingChecks = () => {
  const signals: AbortSignal[] = [];
  vi.mocked(context.actions.getBoardMediaUsage)
    .mockReset()
    .mockImplementation((_boardId, signal) => {
      signals.push(signal);
      return new Promise(() => {});
    });

  return signals;
};

const closeDialog = {
  cancel: () => act(() => userEvent.click(dialogButton('Cancel')!)),
  escape: () => act(() => userEvent.keyboard('{Escape}')),
  // A refetch finds the board gone (deleted elsewhere, or access revoked): the menu closes the dialog on its own.
  external: () => renderMenuFor(board as GalleryBoard, { gallery: { ...context.gallery, boards: [] } } as never),
};

it.each(Object.keys(closeDialog) as (keyof typeof closeDialog)[])(
  'stops a running check when the dialog closes (%s)',
  async (how) => {
    const signals = hangingChecks();
    const dialog = await openDeleteDialog();

    await tickDeleteMedia(dialog);
    await vi.waitFor(() => expect(signals).toHaveLength(1));
    expect(signals[0]!.aborted).toBe(false);

    await closeDialog[how]();
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
    expect(signals[0]!.aborted).toBe(true);
  }
);

it('stops a running check when the media is unticked, and asks again when it is ticked once more', async () => {
  const signals = hangingChecks();
  const dialog = await openDeleteDialog();

  await tickDeleteMedia(dialog);
  await vi.waitFor(() => expect(signals).toHaveLength(1));
  await tickDeleteMedia(dialog);
  expect(signals[0]!.aborted).toBe(true);

  // The stopped check left no answer behind: ticking again within the same opening asks again.
  await tickDeleteMedia(dialog);
  await vi.waitFor(() => expect(signals).toHaveLength(2));
  expect(signals[1]!.aborted).toBe(false);
  expect(dialog.textContent).toContain('Checking where this media is used…');
});

/** Opens the dialog for `menuBoard`, ticks the media and answers the check with `checkedCount` items to delete. */
const answerWithCheckedCount = async (menuBoard: GalleryBoard, checkedCount: number, onlyImages = true) => {
  let settle: (report: GalleryMediaUsageReport) => void = () => {};
  vi.mocked(context.actions.getBoardMediaUsage)
    .mockReset()
    .mockImplementation(
      () =>
        new Promise((resolve) => {
          settle = resolve;
        })
    );
  await renderMenuFor(menuBoard);
  const dialog = await openDeleteDialog();

  await tickDeleteMedia(dialog);
  const before = dialog.textContent;
  const regionBefore = dialog.querySelector<HTMLElement>('[aria-live="polite"]');
  const regionBeforePosition = regionBefore ? getComputedStyle(regionBefore).position : null;
  await act(() =>
    settle({
      ...boardUsageReport,
      checkedCount,
      groups: checkedCount === 0 ? [] : boardUsageReport.groups,
      inUseCount: Math.min(1, checkedCount),
      onlyImages,
    })
  );
  await vi.waitFor(() => expect(dialog.textContent).not.toContain('Checking where this media is used…'), REPORT_WAIT);

  return { before, dialog, regionBefore, regionBeforePosition };
};

it('names only the caller’s own items once the check says only those go, and keeps saying so when unticked', async () => {
  // A shared board of 61 items; the caller added two of them.
  const shared = { ...board, assetCount: 40, imageCount: 20 } as GalleryBoard;
  const { before, dialog } = await answerWithCheckedCount(shared, 2);

  // Until the check answers, the board's own counts are all there is, and so many items need the name typed.
  expect(before).toContain('Its 20 images, 1 videos and 40 uploads are deleted with it.');
  expect(before).toContain('Type "Board 1" to confirm');

  expect(dialog.textContent).toContain('Your 2 images on it are deleted with it; the rest move to Uncategorized.');
  expect(dialog.textContent).toContain('Also permanently delete your 2 items');
  expect(dialog.textContent).toContain('1 of the 2 images this deletes are still in use:');
  // Two items are not many: no name to type.
  expect(dialog.textContent).not.toContain('to confirm');
  expect(dialogButton('Delete board and your 2 items')!.disabled).toBe(false);

  await tickDeleteMedia(dialog);
  expect(dialog.textContent).toContain('Also permanently delete your 2 items');
  await tickDeleteMedia(dialog);
  await act(() => userEvent.click(dialogButton('Delete board and your 2 items')!));
  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

it('keeps the name field, and what was typed, once typing started, however the answer lowers the count', async () => {
  let settle: (report: GalleryMediaUsageReport) => void = () => {};
  vi.mocked(context.actions.getBoardMediaUsage)
    .mockReset()
    .mockImplementation(
      () =>
        new Promise((resolve) => {
          settle = resolve;
        })
    );
  await renderMenuFor({ ...board, assetCount: 40, imageCount: 20 } as GalleryBoard);
  let dialog = await openDeleteDialog();
  const nameField = () => dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])');

  await tickDeleteMedia(dialog);
  await act(() => userEvent.type(nameField()!, 'Boa'));
  await act(() => settle({ ...boardUsageReport, checkedCount: 2, inUseCount: 1, onlyImages: true }));
  await vi.waitFor(() => expect(dialog.textContent).toContain('Also permanently delete your 2 items'), REPORT_WAIT);

  // Two items would not ask for the name, but the user had started: the field and its requirement stay.
  expect(nameField()?.value).toBe('Boa');
  expect(dialogButton('Delete board and your 2 items')!.disabled).toBe(true);
  await act(() => userEvent.type(nameField()!, 'rd 1'));
  expect(dialogButton('Delete board and your 2 items')!.disabled).toBe(false);

  // A new opening starts over: the answer again decides.
  await act(() => userEvent.click(dialogButton('Cancel')!));
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
  dialog = await openDeleteDialog();
  await tickDeleteMedia(dialog);
  await act(() => settle({ ...boardUsageReport, checkedCount: 2, inUseCount: 1, onlyImages: true }));
  await vi.waitFor(() => expect(dialog.textContent).toContain('Also permanently delete your 2 items'), REPORT_WAIT);
  expect(nameField()).toBeNull();
});

it('leaves Retry in place but disabled as the dialog closes', async () => {
  vi.mocked(context.actions.getBoardMediaUsage).mockReset().mockRejectedValue(new Error('offline'));
  const dialog = await openDeleteDialog();

  await tickDeleteMedia(dialog);
  await vi.waitFor(() => expect(dialogButton('Retry')?.disabled).toBe(false), REPORT_WAIT);

  await act(() => userEvent.click(dialogButton('Cancel')!));
  // Still on screen while it animates out, Retry can no longer run.
  expect(dialogButton('Retry')?.disabled).toBe(true);
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
  expect(context.actions.getBoardMediaUsage).toHaveBeenCalledOnce();
});

it('says none of the media goes when the caller added none of it, with no count to delete', async () => {
  const { before, dialog, regionBefore, regionBeforePosition } = await answerWithCheckedCount(board as GalleryBoard, 0);

  // Until the check answers, it shows as running.
  expect(before).toContain('Checking where this media is used…');

  expect(dialog.textContent).toContain('None of its media is yours; all of it moves to Uncategorized.');
  expect(dialog.textContent).toContain('Also permanently delete your items');
  expect(dialog.textContent).not.toMatch(/\b0 (items|images)\b/);
  // Nothing is deleted, so there is no use to report: no "unused" verdict about the board's media. The live region
  // that said it was checking reads out the outcome instead, the description, not shown a second time.
  expect(dialog.textContent).not.toContain('Nothing uses this media.');
  const liveRegion = dialog.querySelector<HTMLElement>('[aria-live="polite"]');
  // The very region that showed the check running, visibly, then.
  expect(liveRegion).toBe(regionBefore);
  expect(regionBeforePosition).not.toBe('absolute');
  expect(liveRegion?.getAttribute('aria-busy')).toBe('false');
  expect(liveRegion?.textContent).toBe('None of its media is yours; all of it moves to Uncategorized.');
  // Hidden as a whole, out of the layout: no gap where it stands.
  expect(getComputedStyle(liveRegion!).position).toBe('absolute');
  expect(liveRegion!.getBoundingClientRect().width).toBeLessThanOrEqual(1);
  await act(() => userEvent.click(dialogButton('Delete board')!));
  // Nothing of the caller's to delete: the board goes without asking for its media.
  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', false);
});

it('keeps counting every item for an administrator overriding, who deletes them all', async () => {
  const overridden = {
    ...board,
    access: {
      actions: ['read', 'rename', 'archive', 'delete'],
      adminOverride: true,
      generalAccess: 'restricted',
      role: null,
    },
  } as GalleryBoard;
  // The board's counts went stale: one item left since they were read.
  const { dialog } = await answerWithCheckedCount(overridden, 2, false);

  expect(dialog.textContent).toContain('Its 2 items are deleted with it.');
  expect(dialog.textContent).toContain('Also permanently delete its 2 items');
  expect(dialog.textContent).not.toContain('your');
  expect(dialogButton('Delete board and 2 items')).toBeDefined();
});

/** The panel's part: the menu target clears when the menu closes. */
const ControlledMenu = ({ initialTarget }: { initialTarget: typeof target }) => {
  const [menuTarget, setMenuTarget] = useState<typeof target | null>(initialTarget);
  const close = useCallback(() => setMenuTarget(null), []);

  return <GalleryBoardMenu target={menuTarget} onClose={close} />;
};

const openMenus = () => document.querySelectorAll('[role="menu"][data-state="open"]');

it.each(['cancel', 'delete'] as const)(
  'closes the board menu as its delete dialog opens, and keeps it closed after %s',
  async (outcome) => {
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <GalleryWidgetContext value={context}>
              <ControlledMenu initialTarget={target} />
            </GalleryWidgetContext>
          </QueryClientProvider>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    expect(openMenus()).toHaveLength(1);

    await openDeleteDialog();
    await vi.waitFor(() => expect(openMenus()).toHaveLength(0));

    await act(() => userEvent.click(dialogButton(outcome === 'cancel' ? 'Cancel' : 'Delete board')!));
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull());

    expect(openMenus()).toHaveLength(0);
    expect(document.activeElement?.closest('[role="menu"]')).toBeNull();
  }
);

it('offers no media choice for an empty board', async () => {
  await renderMenuFor({ ...board, imageCount: 0, videoCount: 0 } as GalleryBoard);
  const dialog = await openDeleteDialog();

  expect(dialog.textContent).toContain('The board is empty.');
  expect(dialog.querySelector('input[type="checkbox"]')).toBeNull();
});

it('makes the board the auto-add destination', async () => {
  const autoAddItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(autoAddItem?.textContent).toBe('widgets.gallery.autoAddToBoard');
  await act(() => userEvent.click(autoAddItem!));

  expect(context.actions.updateSettings).toHaveBeenCalledExactlyOnceWith({
    autoAddBoardId: 'board-1',
    followSelectedBoard: false,
  });
});

it('hands results back to the default destination from the auto-add board itself', async () => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <GalleryWidgetContext value={targetContext}>
            <GalleryBoardMenu target={target} onClose={noop} />
          </GalleryWidgetContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
    await Promise.resolve();
  });

  const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(item?.textContent).toBe('widgets.gallery.stopAutoAdd');
  await act(() => userEvent.click(item!));

  expect(context.actions.updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: null, followSelectedBoard: false });
});

it('offers neither choice on the default destination, where results already go', async () => {
  await renderMenuFor(board, {
    gallery: { ...context.gallery, projectBoardId: 'board-1' } as GalleryWidgetContextValue['gallery'],
  });

  expect(document.querySelector('[role="menuitem"][data-value="auto-add-board"]')).toBeNull();
});

const renderMenuFor = async (menuBoard: GalleryBoard, overrides: Partial<GalleryWidgetContextValue> = {}) => {
  const value = {
    ...context,
    gallery: { ...context.gallery, boards: [menuBoard] },
    ...overrides,
  } as GalleryWidgetContextValue;
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <GalleryWidgetContext value={value}>
            <GalleryBoardMenu target={{ board: menuBoard, x: 20, y: 20 }} onClose={noop} />
          </GalleryWidgetContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const projectGroupElement = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="group"]')).find(
    (element) => element.querySelector('[data-part="item-group-label"]')?.textContent === 'Projects'
  );

const projectGroup = () => {
  const group = projectGroupElement();

  return group
    ? {
        label: 'Projects',
        items: Array.from(group.querySelectorAll('[role="menuitem"]')).map((item) => item.textContent),
      }
    : undefined;
};

const libraryItem = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Library'
  );

const menuItemLabels = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).map((item) => item.textContent);

it('offers every tier but the one the board is in, and moves it there', async () => {
  await renderMenuFor({ ...board, projectId: 'p1' } as GalleryBoard);

  const trigger = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Move to…'
  )!;
  await act(() => userEvent.click(trigger));

  await vi.waitFor(() => {
    expect(menuItemLabels()).toContain('Harbor Tower');
  });
  // In a project: the Library and the other projects, never the project it is already in.
  expect(
    menuItemLabels().filter((label) => label === 'Library' || label === 'Harbor Tower' || label === 'Mahogany House')
  ).toEqual(['Library', 'Harbor Tower']);
  // The Library stands apart, above the projects' heading.
  expect(projectGroup()).toEqual({ label: 'Projects', items: ['Harbor Tower'] });
  expect(libraryItem()?.closest('[role="group"]')).toBeNull();
  expect(libraryItem()?.nextElementSibling?.getAttribute('role')).toBe('separator');

  const library = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Library'
  )!;
  await act(() => userEvent.click(library));

  expect(context.actions.moveBoard).toHaveBeenCalledExactlyOnceWith('board-1', null, 'Library');
});

it('offers a Library board the open project first', async () => {
  await renderMenuFor(board);

  const trigger = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Move to…'
  )!;
  await act(() => userEvent.click(trigger));

  await vi.waitFor(() => {
    expect(menuItemLabels()).toContain('Harbor Tower');
  });
  expect(
    menuItemLabels().filter((label) => label === 'Library' || label === 'Harbor Tower' || label === 'Mahogany House')
  ).toEqual(['Mahogany House', 'Harbor Tower']);
  expect(projectGroup()).toEqual({ label: 'Projects', items: ['Mahogany House', 'Harbor Tower'] });
  expect(libraryItem()).toBeUndefined();
  expect(projectGroupElement()?.previousElementSibling?.getAttribute('role')).not.toBe('separator');
});

it('leaves an inbox with only the actions its project does not own', async () => {
  await renderMenuFor({ ...board, isInbox: true, projectId: 'p1' } as GalleryBoard);

  const labels = menuItemLabels();

  expect(labels).toContain('widgets.gallery.exportProjectFromBoard');
  expect(labels).not.toContain('Move to…');
  expect(labels).not.toContain('Delete board');
  expect(labels).not.toContain('widgets.gallery.renameBoard');
});

const withAccess = (
  role: GalleryBoardAccess['role'],
  actions: GalleryBoardAction[],
  adminOverride = false
): GalleryBoard =>
  ({
    ...board,
    access: { actions, adminOverride, generalAccess: 'restricted', role },
    ownerName: 'Ana',
  }) as GalleryBoard;
const ALL_ACTIONS: GalleryBoardAction[] = [
  'read',
  'add_media',
  'manage_items',
  'rename',
  'set_cover',
  'archive',
  'set_general_access',
  'move',
  'delete',
];

it('offers a viewer only what they may do, and says whose board it is and whom to ask', async () => {
  await renderMenuFor(withAccess('viewer', ['read']), { boardSharing: true });

  expect(menuItemLabels()).toEqual(['Download board (1 video omitted)']);
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.viewer');
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.askToAdd');
});

it('lets an editor rename and archive someone else’s board, but not move, share or delete it', async () => {
  await renderMenuFor(withAccess('editor', ['read', 'add_media', 'manage_items', 'rename', 'set_cover', 'archive']), {
    boardSharing: true,
  });

  expect(menuItemLabels()).toEqual([
    'widgets.gallery.autoAddToBoard',
    'Download board (1 video omitted)',
    'widgets.gallery.renameBoard',
    'widgets.gallery.archiveBoard',
  ]);
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.editor');
  expect(document.body.textContent).not.toContain('askToAdd');
});

it('lets an owner change who can open the board, only where boards can be shared', async () => {
  await renderMenuFor(withAccess('owner', ALL_ACTIONS), { boardSharing: false });
  expect(menuItemLabels().some((label) => label?.includes('widgets.gallery.boardAccess.menu'))).toBe(false);
  expect(document.body.textContent).not.toContain('boardAccess.header');

  await renderMenuFor(withAccess('owner', ALL_ACTIONS), { boardSharing: true });
  const access = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((item) =>
    item.textContent?.includes('widgets.gallery.boardAccess.menu')
  )!;
  await act(() => userEvent.click(access));
  const everyone = await vi.waitFor(() => {
    const option = document.querySelector<HTMLElement>('[role="menuitemradio"][data-value="viewer"]');
    expect(option).not.toBeNull();
    return option!;
  });
  await act(() => userEvent.click(everyone));

  expect(context.actions.setBoardGeneralAccess).toHaveBeenCalledExactlyOnceWith('board-1', 'viewer');
});

it('gives an admin someone else’s board to manage, but no move into their own projects', async () => {
  await renderMenuFor(withAccess(null, ALL_ACTIONS, true), { boardSharing: true });

  const labels = menuItemLabels();
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.admin');
  expect(labels).toContain('Delete board');
  expect(labels).not.toContain('Move to…');
});

it('closes the rename dialog once the board no longer allows renaming', async () => {
  const editable = withAccess('editor', ['read', 'rename']);
  await renderMenuFor(editable);
  const rename = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (item) => item.textContent === 'widgets.gallery.renameBoard'
  )!;
  await act(() => userEvent.click(rename));
  await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).not.toBeNull());

  // A refetch reports the role revoked to viewer.
  await renderMenuFor(editable, {
    gallery: { ...context.gallery, boards: [withAccess('viewer', ['read'])] } as GalleryWidgetContextValue['gallery'],
  });

  await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());

  // The role comes back on a later refetch: the dialog stays closed rather than reopening with what was typed.
  await renderMenuFor(editable, {
    gallery: { ...context.gallery, boards: [editable] } as GalleryWidgetContextValue['gallery'],
  });
  await new Promise((resolve) => {
    setTimeout(resolve, 100);
  });
  expect(document.querySelector('[role="dialog"]')).toBeNull();
});
