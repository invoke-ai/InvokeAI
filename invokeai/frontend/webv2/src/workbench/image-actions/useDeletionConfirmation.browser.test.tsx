import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageReport } from '@features/gallery/mediaUsage';
import type * as projectsApi from '@workbench/projects/api';
import type { Ref } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { RequestDeletionConfirmation } from './useDeletionConfirmation';

import { useDeletionConfirmation } from './useDeletionConfirmation';

// The preference store is real; only its backend write is stubbed.
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const count = Number(values?.count ?? 0);
      const messages: Record<string, string> = {
        'common.cancel': 'Cancel',
        'common.retry': 'Retry',
        'widgets.gallery.deleteBlockedByPending': `${String(values?.media)} not deleted: answer the open deletion first.`,
        'widgets.gallery.deleteBlockedByRunning': `${String(values?.media)} not deleted: wait for the deletion under way.`,
        'widgets.gallery.deleteConfirmLabel': 'Delete',
        'widgets.gallery.deleteConfirmSkip': "Don't ask me again",
        'widgets.gallery.deleteConfirmSkipHint': 'You can turn this confirmation back on in Settings.',
        'widgets.gallery.deleteImagesConfirmBody':
          count === 1 ? 'Permanently delete the image.' : 'Permanently delete these images.',
        'widgets.gallery.deleteImagesConfirmTitle': count === 1 ? 'Delete image?' : `Delete ${count} images?`,
        'widgets.gallery.deleteItemsConfirmBody':
          count === 1 ? 'Permanently delete the item.' : 'Permanently delete these items.',
        'widgets.gallery.deleteItemsConfirmTitle': count === 1 ? 'Delete item?' : `Delete ${count} items?`,
        'widgets.gallery.deleteInUse.checking': 'Checking where this media is used…',
        'widgets.gallery.deleteInUse.failed': 'Couldn’t check where this media is used.',
        'widgets.gallery.deleteInUse.group.openProject': 'This project',
        'widgets.gallery.deleteInUse.group.project': `Project “${String(values?.name)}”`,
        'widgets.gallery.deleteInUse.group.projectUnnamed': 'Untitled project',
        'widgets.gallery.deleteInUse.group.queue':
          count === 1 ? '1 queued or running job' : `${count} queued or running jobs`,
        'widgets.gallery.deleteInUse.group.workflow': `Library workflow “${String(values?.name)}”`,
        'widgets.gallery.deleteInUse.listLabel': 'Where the media is used',
        'widgets.gallery.deleteInUse.none': 'Nothing uses this media.',
        'widgets.gallery.deleteInUse.projectUncheckedHint': 'Couldn’t open it to see where.',
        'widgets.gallery.deleteInUse.savedCopyUncheckedHint': 'Its saved version couldn’t be checked.',
        'widgets.gallery.deleteInUse.queueWarning':
          count === 1 ? 'Jobs that use it will fail.' : 'Jobs that use them will fail.',
        'widgets.gallery.deleteInUse.summary': `${count} of these ${String(values?.media)} are still in use:`,
        'widgets.gallery.deleteInUse.summaryAll': `All ${String(values?.media)} are still in use:`,
        'widgets.gallery.deleteInUse.summaryOne': 'Still in use:',
        'widgets.gallery.deleteInUse.surface.canvasRaster': 'Canvas raster layers',
        'widgets.gallery.deleteInUse.surface.workflow': `Workflow “${String(values?.name)}”`,
        'widgets.gallery.deleteInUse.unchecked': 'Other projects couldn’t be checked.',
        'widgets.gallery.deleteInUse.itemCount': count === 1 ? '1 item' : `${count} items`,
        'widgets.gallery.deleteInUse.warning':
          count === 1
            ? 'Work that uses it will point at missing media.'
            : 'Work that uses them will point at missing media.',
        'widgets.gallery.deleteTargetImage': `Image ${String(values?.index)}`,
        'widgets.gallery.deleteTargetVideo': `Video ${String(values?.index)}`,
        'widgets.gallery.deleteTargetsMore': `+${count}`,
        'widgets.gallery.deleteTargetsMoreLabel': `${count} more items`,
        'widgets.gallery.deleteTargetsSummary': `Deleting ${String(values?.media)}`,
        'widgets.gallery.imageCount': count === 1 ? '1 image' : `${count} images`,
        'widgets.gallery.mediaCountsTwo': `${String(values?.first)} and ${String(values?.second)}`,
        'widgets.gallery.videoCount': count === 1 ? '1 video' : `${count} videos`,
      };

      return messages[key] ?? key;
    },
  }),
}));

/** The report's code loads lazily on the first settled check; a cold transform under a busy run can take a while. */
const REPORT_WAIT = { timeout: 5000 };

type ConfirmationHandle = { requestDeletionConfirmation: RequestDeletionConfirmation };

const Probe = ({ ref }: { ref: Ref<ConfirmationHandle> }) => {
  const { dialog, requestDeletionConfirmation } = useDeletionConfirmation();

  useImperativeHandle(ref, () => ({ requestDeletionConfirmation }), [requestDeletionConfirmation]);

  return dialog;
};

const imageRefs: GalleryItemRef[] = [{ kind: 'image', name: 'image.png' }];
const usageReport = (overrides: Partial<GalleryMediaUsageReport>): GalleryMediaUsageReport => ({
  checkedCount: 1,
  groups: [],
  inUseCount: 0,
  onlyImages: true,
  scope: 'items',
  serverCheck: 'complete',
  ...overrides,
});
const mixedRefs: GalleryItemRef[] = [
  { kind: 'image', name: 'image.png' },
  { kind: 'video', name: 'video.mp4' },
];

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const confirmationRef = createRef<ConfirmationHandle>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderProbe = async () => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <Probe ref={confirmationRef} />
      </ChakraProvider>
    )
  );
};

const clickButton = async (label: string) => {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === label);

  if (!button) {
    throw new Error(`Button "${label}" did not render.`);
  }

  await act(async () => {
    button.click();
    await Promise.resolve();
  });
};

const openDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"][data-state="open"]');
const skipCheckbox = () => openDialog()?.querySelector<HTMLInputElement>('input[type="checkbox"]') ?? null;
const skipLabel = () => openDialog()?.querySelector<HTMLElement>('[data-scope="checkbox"][data-part="label"]') ?? null;
const buttonNamed = (label: string) =>
  Array.from(openDialog()?.querySelectorAll('button') ?? []).find((button) => button.textContent === label) ?? null;

beforeEach(async () => {
  accountLifecycle.activate('deletion-confirmation-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderProbe();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
});

describe('useDeletionConfirmation', () => {
  it('waits for confirmation and keeps the dialog pending through deletion', async () => {
    let resolveDeletion!: () => void;
    const executeDeletion = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveDeletion = resolve;
        })
    );
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion);
    });

    expect(document.querySelector('[role="alertdialog"][data-state="open"]')?.textContent).toContain('Delete image?');
    expect(executeDeletion).not.toHaveBeenCalled();

    await clickButton('Delete');
    expect(executeDeletion).toHaveBeenCalledOnce();
    expect(document.querySelector('[role="alertdialog"][data-state="open"]')).not.toBeNull();

    await act(async () => {
      resolveDeletion();
      await request;
    });

    expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull();
    // Confirming without the skip option leaves confirmation on.
    expect(getWorkbenchPreferences().confirmImageDeletion).toBe(true);
  });

  it('opens at once while it checks where the items are used, then lists each place grouped by owner', async () => {
    const executeDeletion = vi.fn(async () => {});
    const consoleError = vi.spyOn(console, 'error');
    let settleCheck: (report: GalleryMediaUsageReport) => void = () => {};
    const checkUsage = vi.fn(
      () =>
        new Promise<GalleryMediaUsageReport>((resolve) => {
          settleCheck = resolve;
        })
    );

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(mixedRefs, executeDeletion, {
        canSkipFutureConfirmations: true,
        checkUsage,
      });
    });

    expect(openDialog()?.textContent).toContain('Checking where this media is used…');
    // Deleting anyway does not wait for the check.
    expect(buttonNamed('Delete')?.disabled).toBe(false);

    await act(() =>
      settleCheck(
        usageReport({
          checkedCount: 2,
          groups: [
            {
              count: 2,
              kind: 'openProject',
              surfaces: [
                { count: 1, surface: 'canvasRaster' },
                { count: 2, label: 'Upscale pass', surface: 'workflow' },
                // Workflow names are not unique.
                { count: 1, label: 'Upscale pass', surface: 'workflow' },
              ],
            },
            { count: 1, kind: 'projectUnchecked', name: 'Harbor Tower' },
            // Open in this tab, its saved version unread: what it uses here, and a note.
            { count: 1, kind: 'project', name: 'Atlas', savedCopyUnchecked: true, surfaces: [] },
            { count: 1, kind: 'workflow', name: 'Portraits' },
            { count: 1, jobCount: 2, kind: 'queue' },
          ],
          inUseCount: 2,
          onlyImages: false,
        })
      )
    );

    // The section loads with the first check a dialog shows.
    await vi.waitFor(() => expect(openDialog()?.querySelector('[role="group"]')).not.toBeNull(), REPORT_WAIT);
    const dialog = openDialog();
    const group = dialog?.querySelector('[role="group"]');
    expect(dialog?.querySelector(`#${group?.getAttribute('aria-labelledby')}`)?.textContent).toBe(
      'All 2 items are still in use:'
    );
    const rows = Array.from(group?.querySelectorAll('li') ?? []).map((row) =>
      Array.from(row.querySelectorAll(':scope > div > p'))
        .map((cell) => cell.textContent)
        .join(' · ')
    );
    expect(rows).toEqual([
      'This project · 2 items',
      'Canvas raster layers · 1 item',
      'Workflow “Upscale pass” · 2 items',
      'Workflow “Upscale pass” · 1 item',
      'Project “Harbor Tower” · 1 item',
      'Project “Atlas” · 1 item',
      'Library workflow “Portraits” · 1 item',
      '2 queued or running jobs · 1 item',
    ]);
    expect(dialog?.textContent).toContain('Couldn’t open it to see where.');
    expect(dialog?.textContent).toContain('Its saved version couldn’t be checked.');
    expect(dialog?.textContent).toContain(
      // Two items in use: the warnings speak of them, not it.
      'Work that uses them will point at missing media. Jobs that use them will fail.'
    );
    expect(consoleError).not.toHaveBeenCalledWith(expect.stringContaining('same key'), expect.anything());
    // With routine confirmations on, the opt-out still means something for later deletions.
    expect(skipCheckbox()).not.toBeNull();

    await clickButton('Delete');
    expect(executeDeletion).toHaveBeenCalledOnce();
    consoleError.mockRestore();
  });

  it('warns when the check failed, offers Retry, and still offers the deletion', async () => {
    let settleRetry: (report: GalleryMediaUsageReport) => void = () => {};
    const checkUsage = vi
      .fn<(signal: AbortSignal) => Promise<GalleryMediaUsageReport>>()
      .mockRejectedValueOnce(new Error('offline'))
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            settleRetry = resolve;
          })
      );

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        { checkUsage }
      );
    });
    await vi.waitFor(
      () => expect(openDialog()?.textContent).toContain('Couldn’t check where this media is used.'),
      REPORT_WAIT
    );
    expect(buttonNamed('Delete')?.disabled).toBe(false);

    // From the keyboard: Retry holds focus as Enter runs it. The dialog first settles its own opening focus on Cancel,
    // which would otherwise take focus back from Retry and receive the Enter.
    await vi.waitFor(() => expect(document.activeElement).toBe(buttonNamed('Cancel')));
    buttonNamed('Retry')!.focus();
    expect(document.activeElement).toBe(buttonNamed('Retry'));
    await act(() => userEvent.keyboard('{Enter}'));

    // Retry gives way to the checking state; focus stays in the dialog, on the panel that reads the outcome, and shows
    // the ring keyboard focus does.
    const panel = openDialog()?.querySelector<HTMLElement>('[aria-live="polite"]');
    expect(openDialog()?.textContent).toContain('Checking where this media is used…');
    expect(document.activeElement).toBe(panel);
    expect(panel?.matches(':focus-visible')).toBe(true);
    expect(getComputedStyle(panel!).outlineStyle).not.toBe('none');

    await act(() => settleRetry(usageReport({ inUseCount: 0 })));

    await vi.waitFor(() => expect(openDialog()?.textContent).toContain('Nothing uses this media.'), REPORT_WAIT);
    expect(document.activeElement).toBe(panel);
    expect(openDialog()?.contains(document.activeElement)).toBe(true);
    expect(checkUsage).toHaveBeenCalledTimes(2);
  });

  it('keeps what it already knows on screen while a retry runs, checking where the warning and Retry were', async () => {
    let settleRetry: (report: GalleryMediaUsageReport) => void = () => {};
    const known = usageReport({
      groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
      inUseCount: 1,
      serverCheck: 'unavailable',
    });
    const checkUsage = vi.fn<(signal: AbortSignal) => Promise<GalleryMediaUsageReport>>().mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          settleRetry = resolve;
        })
    );

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          checkUsage,
          usage: known,
        }
      );
    });
    await vi.waitFor(
      () => expect(openDialog()?.textContent).toContain('Other projects couldn’t be checked.'),
      REPORT_WAIT
    );

    await act(() => userEvent.click(buttonNamed('Retry')!));

    const panel = openDialog()?.querySelector<HTMLElement>('[aria-live="polite"]');
    expect(panel?.getAttribute('aria-busy')).toBe('true');
    expect(panel?.textContent).toContain('This project');
    expect(panel?.textContent).toContain('Checking where this media is used…');
    expect(panel?.textContent).not.toContain('Other projects couldn’t be checked.');
    expect(buttonNamed('Retry')).toBeNull();
    expect(document.activeElement).toBe(panel);

    await act(() => settleRetry({ ...known, serverCheck: 'complete' }));

    await vi.waitFor(() => expect(panel?.textContent).not.toContain('Checking'), REPORT_WAIT);
    expect(panel?.textContent).toContain('This project');
    expect(panel?.getAttribute('aria-busy')).toBe('false');
    expect(document.activeElement).toBe(panel);
  });

  it('keeps the earlier answer, not the checking state, when deleted anyway during a retry', async () => {
    let resolveDeletion!: () => void;
    const executeDeletion = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveDeletion = resolve;
        })
    );
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion, {
        checkUsage: () => new Promise(() => {}),
        usage: usageReport({
          groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
          inUseCount: 1,
          serverCheck: 'unavailable',
        }),
      });
    });
    await vi.waitFor(() => expect(buttonNamed('Retry')).not.toBeNull(), REPORT_WAIT);
    await act(() => userEvent.click(buttonNamed('Retry')!));
    expect(openDialog()?.textContent).toContain('Checking where this media is used…');

    await clickButton('Delete');

    // The deletion runs, the dialog with it: the answer the retry was replacing is back, with no spinner, and its Retry
    // stays in place but cannot run.
    expect(executeDeletion).toHaveBeenCalledOnce();
    await vi.waitFor(() => expect(openDialog()?.textContent).toContain('Other projects couldn’t be checked.'));
    expect(openDialog()?.textContent).toContain('This project');
    expect(openDialog()?.textContent).not.toContain('Checking where this media is used…');
    expect(buttonNamed('Retry')?.disabled).toBe(true);

    await act(async () => {
      resolveDeletion();
      await request;
    });
  });

  it('shows a known report at once, without a checking state, and says what went unchecked', async () => {
    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          checkUsage: () => new Promise(() => {}),
          usage: usageReport({
            groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
            inUseCount: 1,
            serverCheck: 'unavailable',
          }),
        }
      );
    });

    expect(openDialog()?.textContent).not.toContain('Checking');
    await vi.waitFor(() => expect(openDialog()?.textContent).toContain('Still in use:'), REPORT_WAIT);
    expect(openDialog()?.textContent).toContain('Other projects couldn’t be checked.');
    // Deleting one item, every row would say "1 item": the rows name places only.
    expect(Array.from(openDialog()?.querySelectorAll('[role="group"] li') ?? []).map((row) => row.textContent)).toEqual(
      ['This projectCanvas raster layers', 'Canvas raster layers']
    );
  });

  it('announces the outcome in a region present from the moment the dialog opens, beside its description', async () => {
    let settle: (report: GalleryMediaUsageReport) => void = () => {};

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          checkUsage: () =>
            new Promise((resolve) => {
              settle = resolve;
            }),
        }
      );
    });

    const region = openDialog()?.querySelector('[aria-live="polite"]');
    expect(region?.getAttribute('aria-busy')).toBe('true');
    expect(region?.textContent).toBe('Checking where this media is used…');
    const describedBy = openDialog()?.getAttribute('aria-describedby');
    expect(describedBy && document.getElementById(describedBy)?.textContent).toBe('Permanently delete the image.');

    await act(() => settle(usageReport({ inUseCount: 0 })));

    // The same element, so assistive technology reads the change in it.
    await vi.waitFor(() => expect(region?.textContent).toBe('Nothing uses this media.'), REPORT_WAIT);
    expect(region?.isConnected).toBe(true);
    expect(region?.getAttribute('aria-busy')).toBe('false');
  });

  it("keeps an earlier request's late answer out of a later dialog", async () => {
    const settles: Array<(report: GalleryMediaUsageReport) => void> = [];
    const checkUsage = () =>
      new Promise<GalleryMediaUsageReport>((resolve) => {
        settles.push(resolve);
      });
    const inUse = usageReport({
      groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
      inUseCount: 1,
    });

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        { checkUsage }
      );
    });
    await clickButton('Cancel');
    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        { checkUsage }
      );
    });

    await act(() => settles[0]?.(inUse));
    expect(openDialog()?.textContent).toContain('Checking where this media is used…');
    expect(openDialog()?.textContent).not.toContain('still in use');

    await act(() => settles[1]?.(usageReport({ inUseCount: 0 })));
    await vi.waitFor(() => expect(openDialog()?.textContent).toContain('Nothing uses this media.'), REPORT_WAIT);
  });

  it('deletes at once when confirmed while the check runs, and stops the check', async () => {
    let resolveDeletion!: () => void;
    const executeDeletion = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveDeletion = resolve;
        })
    );
    const captured: { signal?: AbortSignal } = {};
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion, {
        checkUsage: (signal) => {
          captured.signal = signal;
          return new Promise(() => {});
        },
      });
    });
    expect(openDialog()?.textContent).toContain('Checking where this media is used…');
    await clickButton('Delete');

    expect(executeDeletion).toHaveBeenCalledOnce();
    expect(captured.signal?.aborted).toBe(true);
    // The deletion is still running, the dialog with it; the stopped check no longer says it is checking.
    expect(openDialog()).not.toBeNull();
    expect(openDialog()?.textContent).not.toContain('Checking where this media is used…');

    await act(async () => {
      resolveDeletion();
      await request;
    });
  });

  it('stops the check when the dialog is cancelled', async () => {
    const captured: { signal?: AbortSignal } = {};

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          checkUsage: (signal) => {
            captured.signal = signal;
            return new Promise(() => {});
          },
        }
      );
    });
    await clickButton('Cancel');

    expect(captured.signal?.aborted).toBe(true);
  });

  it('offers no opt-out when routine confirmations are already off and the media is in use', async () => {
    await patchWorkbenchPreferences({ confirmImageDeletion: false });

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          canSkipFutureConfirmations: true,
          usage: usageReport({
            groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'canvasRaster' }] }],
            inUseCount: 1,
          }),
        }
      );
    });

    await vi.waitFor(() => expect(openDialog()?.textContent).toContain('Still in use:'), REPORT_WAIT);
    expect(skipCheckbox()).toBeNull();
    expect(openDialog()?.textContent).not.toContain('You can turn this confirmation back on in Settings.');
  });

  it('offers no opt-out to a request that cannot skip it, such as a batch or an item off screen', async () => {
    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        mixedRefs,
        vi.fn(async () => {})
      );
    });

    expect(openDialog()?.textContent).toContain('Delete 2 items?');
    expect(skipCheckbox()).toBeNull();
  });

  it('shows what it deletes: the count of each kind, the first thumbnails and how many more', async () => {
    const refs: GalleryItemRef[] = [
      ...Array.from({ length: 9 }, (_, index) => ({ kind: 'image' as const, name: `image-${index}.png` })),
      { kind: 'video', name: 'clip.mp4' },
    ];

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        refs,
        vi.fn(async () => {})
      );
    });

    const targets = openDialog()?.querySelector('[role="list"]');
    expect(openDialog()?.textContent).toContain('Deleting 9 images and 1 video');
    // One row: seven thumbnails and a tile counting the other three.
    expect(Array.from(targets?.querySelectorAll('img') ?? []).map((image) => image.getAttribute('alt'))).toEqual(
      Array.from({ length: 7 }, (_, index) => `Image ${index + 1}`)
    );
    expect(targets?.querySelector('img')?.getAttribute('src')).toContain('/images/i/image-0.png/thumbnail');
    expect(targets?.lastElementChild?.getAttribute('aria-label')).toBe('3 more items');
    expect(targets?.lastElementChild?.textContent).toBe('+3');
  });

  it('shows no usage section when the request asks for no check', async () => {
    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {})
      );
    });

    expect(openDialog()?.querySelector('[role="group"]')).toBeNull();
  });

  it('cancels without executing deletion', async () => {
    const executeDeletion = vi.fn(async () => {});
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(mixedRefs, executeDeletion);
    });
    expect(document.querySelector('[role="alertdialog"][data-state="open"]')?.textContent).toContain('Delete 2 items?');

    await clickButton('Cancel');
    await request;

    expect(executeDeletion).not.toHaveBeenCalled();
    expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull();
  });

  it('treats Escape as cancellation without executing deletion', async () => {
    const executeDeletion = vi.fn(async () => {});
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion);
    });
    await vi.waitFor(() =>
      expect(document.querySelector('[role="alertdialog"][data-state="open"]')?.contains(document.activeElement)).toBe(
        true
      )
    );
    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull());
    await request;

    expect(executeDeletion).not.toHaveBeenCalled();
    expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull();
  });

  it('refuses a second request while confirmation is pending, and says it did not delete', async () => {
    const createToast = vi.spyOn(toaster, 'create');
    const firstDeletion = vi.fn(async () => {});
    const secondDeletion = vi.fn(async () => {});

    try {
      await act(() => {
        void confirmationRef.current!.requestDeletionConfirmation(imageRefs, firstDeletion);
        // With routine confirmations off, a second item whose check found it in use arrives on its own, much later.
        void confirmationRef.current!.requestDeletionConfirmation(mixedRefs, secondDeletion);
      });

      expect(createToast).toHaveBeenCalledExactlyOnceWith({
        title: '1 image and 1 video not deleted: answer the open deletion first.',
        type: 'warning',
      });
      expect(document.querySelector('[role="alertdialog"][data-state="open"]')?.textContent).toContain('Delete image?');
      await clickButton('Delete');

      expect(firstDeletion).toHaveBeenCalledOnce();
      expect(secondDeletion).not.toHaveBeenCalled();
    } finally {
      createToast.mockRestore();
    }
  });

  it('says a second request waits for a confirmed deletion still running, not for an answer', async () => {
    const createToast = vi.spyOn(toaster, 'create');
    let resolveDeletion!: () => void;
    const firstDeletion = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveDeletion = resolve;
        })
    );
    let request!: Promise<void>;

    try {
      await act(() => {
        request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, firstDeletion);
      });
      await clickButton('Delete');
      await act(() => {
        void confirmationRef.current!.requestDeletionConfirmation(
          mixedRefs,
          vi.fn(async () => {})
        );
      });

      expect(createToast).toHaveBeenCalledExactlyOnceWith({
        title: '1 image and 1 video not deleted: wait for the deletion under way.',
        type: 'warning',
      });

      await act(async () => {
        resolveDeletion();
        await request;
      });
    } finally {
      createToast.mockRestore();
    }
  });

  it('settles a pending request safely when the host unmounts', async () => {
    const executeDeletion = vi.fn(async () => {});
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion);
    });
    await act(() => root?.unmount());
    await request;

    expect(executeDeletion).not.toHaveBeenCalled();
  });

  it('opens on Cancel, with the skip option before it and Delete after it in tab order', async () => {
    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        { canSkipFutureConfirmations: true }
      );
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(buttonNamed('Cancel')));

    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(buttonNamed('Delete'));

    await act(() => userEvent.tab({ shift: true }));
    await act(() => userEvent.tab({ shift: true }));
    expect(document.activeElement).toBe(skipCheckbox());
    expect(skipCheckbox()?.getAttribute('aria-describedby')).toBeTruthy();
    expect(document.getElementById(skipCheckbox()!.getAttribute('aria-describedby')!)?.textContent).toContain(
      'Settings'
    );
  });

  it('turns confirmation off only when the deletion it was ticked for is confirmed', async () => {
    const executeDeletion = vi.fn(async () => {});

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion, {
        canSkipFutureConfirmations: true,
      });
    });
    await act(() => userEvent.click(skipLabel()!));
    expect(skipCheckbox()?.checked).toBe(true);
    expect(getWorkbenchPreferences().confirmImageDeletion).toBe(true);

    await clickButton('Delete');

    expect(executeDeletion).toHaveBeenCalledOnce();
    expect(getWorkbenchPreferences().confirmImageDeletion).toBe(false);
  });

  it.each(['Cancel', 'Escape', 'the close button'] as const)(
    'leaves confirmation on when a ticked dialog is dismissed by %s',
    async (how) => {
      const executeDeletion = vi.fn(async () => {});
      let request!: Promise<void>;

      await act(() => {
        request = confirmationRef.current!.requestDeletionConfirmation(imageRefs, executeDeletion, {
          canSkipFutureConfirmations: true,
        });
      });
      await act(() => userEvent.click(skipLabel()!));
      expect(skipCheckbox()?.checked).toBe(true);

      if (how === 'Cancel') {
        await clickButton('Cancel');
      } else if (how === 'Escape') {
        await act(() => userEvent.keyboard('{Escape}'));
      } else {
        await act(() => userEvent.click(openDialog()!.querySelector<HTMLElement>('[data-part="close-trigger"]')!));
      }
      await request;

      expect(executeDeletion).not.toHaveBeenCalled();
      expect(getWorkbenchPreferences().confirmImageDeletion).toBe(true);
    }
  );

  it('starts every request with the skip option clear', async () => {
    let request!: Promise<void>;

    await act(() => {
      request = confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          canSkipFutureConfirmations: true,
        }
      );
    });
    await act(() => userEvent.click(skipLabel()!));
    await clickButton('Cancel');
    await request;
    await vi.waitFor(() => expect(openDialog()).toBeNull());

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(
        imageRefs,
        vi.fn(async () => {}),
        {
          canSkipFutureConfirmations: true,
        }
      );
    });

    expect(skipCheckbox()?.checked).toBe(false);
  });
});
