import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useCallback } from 'react';

import { getPreviewComparisonMode, getPreviewFilmstripVisible } from './previewSettings';

const EMPTY = {};

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const read = field.id === 'comparisonMode' ? getPreviewComparisonMode : getPreviewFilmstripVisible;
  const { disabled, layer, set, value } = useLayeredSetting<boolean | string>(
    'preview',
    field.id,
    target,
    read,
    read(EMPTY),
    field
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (field.id === 'comparisonMode' && (next === 'slider' || next === 'side-by-side' || next === 'hover')) {
        set(next);
      } else if (field.id === 'filmstripVisible' && typeof next === 'boolean') {
        set(next);
      }
    },
    [field.id, set]
  );
  return (
    <SettingControl
      layer={layer}
      disabled={disabled}
      field={field}
      surface={surface}
      value={value}
      onChange={onChange}
    />
  );
};
