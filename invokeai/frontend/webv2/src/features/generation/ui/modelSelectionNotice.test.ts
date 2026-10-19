import type { TFunction } from 'i18next';

import { describe, expect, it, vi } from 'vitest';

import { notifyGenerateModelSelectionCleared } from './modelSelectionNotice';

const createTranslation = () =>
  vi.fn((key: string, options?: Record<string, unknown>) => {
    if (key === 'widgets.generate.incompatibleSettingsCleared') {
      return 'Incompatible settings cleared';
    }

    return `${String(options?.labels)} ${Number(options?.count) === 1 ? 'was' : 'were'} not compatible with ${String(options?.name)}.`;
  }) as unknown as TFunction;

describe('notifyGenerateModelSelectionCleared', () => {
  it('does not notify when no settings were cleared', () => {
    const info = vi.fn();

    notifyGenerateModelSelectionCleared({
      clearedLabels: [],
      locale: 'en',
      modelName: 'FLUX',
      notifications: { info },
      t: createTranslation(),
    });

    expect(info).not.toHaveBeenCalled();
  });

  it('passes singular interpolation values to the translated notice', () => {
    const info = vi.fn();
    const t = createTranslation();

    notifyGenerateModelSelectionCleared({
      clearedLabels: ['LoRAs'],
      locale: 'en',
      modelName: 'FLUX',
      notifications: { info },
      t,
    });

    expect(t).toHaveBeenLastCalledWith('widgets.generate.incompatibleSettingsClearedDescription', {
      count: 1,
      labels: 'LoRAs',
      name: 'FLUX',
    });
    expect(info).toHaveBeenCalledWith('Incompatible settings cleared', 'LoRAs was not compatible with FLUX.');
  });

  it('formats multiple labels with the resolved locale and plural interpolation', () => {
    const info = vi.fn();
    const t = createTranslation();

    notifyGenerateModelSelectionCleared({
      clearedLabels: ['Dimensions', 'LoRAs', 'VAE'],
      locale: 'en',
      modelName: 'FLUX',
      notifications: { info },
      t,
    });

    expect(t).toHaveBeenLastCalledWith('widgets.generate.incompatibleSettingsClearedDescription', {
      count: 3,
      labels: 'Dimensions, LoRAs, and VAE',
      name: 'FLUX',
    });
    expect(info).toHaveBeenCalledWith(
      'Incompatible settings cleared',
      'Dimensions, LoRAs, and VAE were not compatible with FLUX.'
    );
  });
});
