import type { GalleryUiAdapter } from '@features/gallery/react';
import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryUiProvider } from '@features/gallery/react';
import { gallerySettingsContribution } from '@features/gallery/settingsContribution';
import { system } from '@theme/system';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GallerySettingField } from './GallerySettingField';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'widgets.gallery.imageSize': 'Image size',
        'widgets.gallery.showPendingItems': 'Show pending items',
      })[key] ?? key,
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const STALE_TARGET = { projectId: 'old-project' };
const PICKED_BOARD_VALUES = { selectedBoardId: 'picked' };
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Harness = ({
  id,
  initialValues = { imageDensityPercent: 25 },
  target,
}: {
  id: string;
  initialValues?: Record<string, unknown>;
  target?: SettingFieldProps['target'];
}) => {
  const [values, setValues] = useState<Record<string, unknown>>(initialValues);
  const updateSettings = useCallback(
    (patch: Record<string, unknown>) => setValues((current) => ({ ...current, ...patch })),
    []
  );
  const adapter = useMemo(
    () => ({ gallery: { updateSettings }, galleryValues: values, projectId: 'project' }) as unknown as GalleryUiAdapter,
    [updateSettings, values]
  );
  const field = gallerySettingsContribution.fields.find((entry) => entry.id === id)!;

  return (
    <ChakraProvider value={system}>
      <GalleryUiProvider adapter={adapter}>
        <div data-surface="quick">
          <GallerySettingField field={field} surface="quick" target={target} />
        </div>
        <div data-surface="dialog">
          <GallerySettingField field={field} surface="dialog" target={target} />
        </div>
        <output>{JSON.stringify(values)}</output>
      </GalleryUiProvider>
    </ChakraProvider>
  );
};

beforeEach(() => {
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

describe('Gallery settings contribution bindings', () => {
  it('inverts persisted density and synchronizes image size across quick and dialog controls', async () => {
    await act(() => root?.render(<Harness id="imageSize" />));
    const quick = host?.querySelector<HTMLElement>('[data-surface="quick"] [role="slider"]');
    const dialog = host?.querySelector<HTMLElement>('[data-surface="dialog"] [role="slider"]');

    expect(quick?.getAttribute('aria-valuenow')).toBe('75');
    expect(dialog?.getAttribute('aria-valuenow')).toBe('75');

    await act(() => quick?.focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));

    expect(host?.querySelector('output')?.textContent).toContain('"imageDensityPercent":24');
    expect(dialog?.getAttribute('aria-valuenow')).toBe('76');
  });

  it('synchronizes toggle changes across presentations through the existing settings command', async () => {
    await act(() => root?.render(<Harness id="showPendingItems" />));
    const quick = host?.querySelector<HTMLInputElement>('[data-surface="quick"] input');
    const dialog = host?.querySelector<HTMLInputElement>('[data-surface="dialog"] input');

    expect(quick?.checked).toBe(true);
    expect(dialog?.checked).toBe(true);
    await act(() => quick?.click());

    expect(dialog?.checked).toBe(false);
    expect(host?.querySelector('output')?.textContent).toContain('"showPendingItems":false');
  });

  it('pins the current destination when results stop following the selected board', async () => {
    await act(() => root?.render(<Harness id="followSelectedBoard" initialValues={PICKED_BOARD_VALUES} />));
    const toggle = host?.querySelector<HTMLInputElement>('[data-surface="quick"] input');

    expect(toggle?.checked).toBe(true);
    await act(() => toggle?.click());

    expect(toggle?.checked).toBe(false);
    expect(host?.querySelector('output')?.textContent).toContain('"autoAddBoardId":"picked"');

    await act(() => toggle?.click());

    expect(host?.querySelector('output')?.textContent).toContain('"autoAddBoardId":"follow"');
  });

  it('disables edits for a target belonging to another project', async () => {
    await act(() => root?.render(<Harness id="showPendingItems" target={STALE_TARGET} />));
    const quick = host?.querySelector<HTMLInputElement>('[data-surface="quick"] input');

    expect(quick?.disabled).toBe(true);
    await act(() => quick?.click());
    expect(host?.querySelector('output')?.textContent).not.toContain('showPendingItems');
  });

  it('applies pagination changes from the quick selector to the dialog selector', async () => {
    await act(() => root?.render(<Harness id="paginationMode" />));
    const trigger = host?.querySelector<HTMLButtonElement>('[data-surface="quick"] [role="combobox"]');
    await act(() => trigger?.click());
    const option = host?.querySelector<HTMLElement>('[data-surface="quick"] [role="option"][data-value="paginated"]');
    expect(option).not.toBeNull();
    await act(() => option?.click());

    expect(host?.querySelector('output')?.textContent).toContain('"paginationMode":"paginated"');
    expect(host?.querySelector('[data-surface="dialog"] [role="combobox"]')?.textContent).toContain('common.pages');
  });
});
