import { ChakraProvider } from '@chakra-ui/react';
import { DEFAULT_GALLERY_SETTINGS } from '@features/gallery/core/settings';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GallerySettingsMenu } from './GallerySettingsMenu';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'widgets.gallery.imageSize': 'Image size',
        'widgets.gallery.settings': 'Settings',
        'widgets.gallery.settingsGridSize': 'Grid Size',
      })[key] ?? key,
  }),
}));

const settings = { ...DEFAULT_GALLERY_SETTINGS, imageDensityPercent: 25 };
const onUpdateSettings = vi.fn();
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderMenu = async () => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GallerySettingsMenu settings={settings} onUpdateSettings={onUpdateSettings} />
      </ChakraProvider>
    )
  );

  const trigger = host?.querySelector<HTMLButtonElement>('button[aria-label="Settings"]');

  await act(async () => {
    trigger?.click();
    await Promise.resolve();
  });

  return trigger;
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  onUpdateSettings.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GallerySettingsMenu grid size', () => {
  it('presents the inverse of persisted density so higher values mean larger items', async () => {
    const trigger = await renderMenu();
    const menuContentId = trigger?.getAttribute('data-controls');
    const menuContent = menuContentId ? document.getElementById(menuContentId) : null;

    const slider = menuContent?.querySelector<HTMLElement>('[role="slider"][aria-label="Image size"]');

    expect(menuContent?.textContent).toContain('Grid Size');
    expect(menuContent?.textContent).toContain('75%');
    expect(slider?.getAttribute('aria-valuenow')).toBe('75');

    slider?.focus();
    await act(() => userEvent.keyboard('{ArrowRight}'));

    expect(onUpdateSettings).toHaveBeenLastCalledWith({ imageDensityPercent: 24 });
  });
});
