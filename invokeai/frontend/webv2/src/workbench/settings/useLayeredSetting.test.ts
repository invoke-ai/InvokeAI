import type { SettingDefinition } from '@platform/ui/settings/contracts';
import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import { formatSettingValue } from './useLayeredSetting';

const t = ((key: string) =>
  ({ 'common.off': 'Off', 'common.on': 'On', 'settingsDialog.layer.automatic': 'Automatic' })[key] ??
  key) as unknown as TFunction;

describe('formatSettingValue', () => {
  const select: SettingDefinition = {
    id: 'thumbnailFit',
    kind: 'select',
    label: 'Thumbnail shape',
    options: [
      { label: 'Square', value: 'square' },
      { label: (translate) => translate('Aspect ratio'), value: 'aspect' },
    ],
    scope: 'project',
  };

  it("names a select value by its option's label, and an unknown one by itself", () => {
    expect(formatSettingValue(select, 'square', t)).toBe('Square');
    expect(formatSettingValue(select, 'aspect', t)).toBe('Aspect ratio');
    expect(formatSettingValue(select, 'circle', t)).toBe('circle');
  });

  it('names booleans On and Off, numbers as written, and no value as Automatic', () => {
    const field: SettingDefinition = { id: 'x', kind: 'boolean', label: 'X', scope: 'project' };
    expect(formatSettingValue(field, true, t)).toBe('On');
    expect(formatSettingValue(field, false, t)).toBe('Off');
    expect(formatSettingValue({ ...field, kind: 'number', max: 2, min: 0 }, 0.25, t)).toBe('0.25');
    expect(formatSettingValue({ ...field, kind: 'custom' }, null, t)).toBe('Automatic');
  });
});
