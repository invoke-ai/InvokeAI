import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useWidgetSettingsTarget } from '@workbench/settings/useWidgetSettingsTarget';
import { useCallback } from 'react';

import { getPreviewComparisonMode, getPreviewFilmstripVisible } from './previewSettings';

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const { disabled, patch, value } = useWidgetSettingsTarget<boolean | string>('preview', target, (values) =>
    field.id === 'comparisonMode' ? getPreviewComparisonMode(values) : getPreviewFilmstripVisible(values)
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (field.id === 'comparisonMode' && (next === 'slider' || next === 'side-by-side' || next === 'hover')) {
        patch({ comparisonMode: next });
      } else if (field.id === 'filmstripVisible' && typeof next === 'boolean') {
        patch({ filmstripVisible: next });
      }
    },
    [field.id, patch]
  );
  const defaultValue = field.id === 'comparisonMode' ? getPreviewComparisonMode({}) : getPreviewFilmstripVisible({});
  return (
    <SettingControl
      isModified={value !== defaultValue}
      disabled={disabled}
      field={field}
      surface={surface}
      value={value}
      onChange={onChange}
    />
  );
};
