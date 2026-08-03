import type { GalleryItemRef } from '@features/gallery/contracts';
import type { Ref } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { RequestDeletionConfirmation } from './useDeletionConfirmation';

import { useDeletionConfirmation } from './useDeletionConfirmation';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const count = Number(values?.count ?? 0);
      const messages: Record<string, string> = {
        'widgets.gallery.deleteConfirmLabel': 'Delete',
        'widgets.gallery.deleteImagesConfirmBody':
          count === 1 ? 'Permanently delete the image.' : 'Permanently delete these images.',
        'widgets.gallery.deleteImagesConfirmTitle': count === 1 ? 'Delete image?' : `Delete ${count} images?`,
        'widgets.gallery.deleteItemsConfirmBody':
          count === 1 ? 'Permanently delete the item.' : 'Permanently delete these items.',
        'widgets.gallery.deleteItemsConfirmTitle': count === 1 ? 'Delete item?' : `Delete ${count} items?`,
      };

      return messages[key] ?? key;
    },
  }),
}));

type ConfirmationHandle = { requestDeletionConfirmation: RequestDeletionConfirmation };

const Probe = ({ ref }: { ref: Ref<ConfirmationHandle> }) => {
  const { dialog, requestDeletionConfirmation } = useDeletionConfirmation();

  useImperativeHandle(ref, () => ({ requestDeletionConfirmation }), [requestDeletionConfirmation]);

  return dialog;
};

const imageRefs: GalleryItemRef[] = [{ kind: 'image', name: 'image.png' }];
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

beforeEach(async () => {
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

  it('ignores a second request while confirmation is pending', async () => {
    const firstDeletion = vi.fn(async () => {});
    const secondDeletion = vi.fn(async () => {});

    await act(() => {
      void confirmationRef.current!.requestDeletionConfirmation(imageRefs, firstDeletion);
      void confirmationRef.current!.requestDeletionConfirmation(mixedRefs, secondDeletion);
    });

    expect(document.querySelector('[role="alertdialog"][data-state="open"]')?.textContent).toContain('Delete image?');
    await clickButton('Delete');

    expect(firstDeletion).toHaveBeenCalledOnce();
    expect(secondDeletion).not.toHaveBeenCalled();
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
});
