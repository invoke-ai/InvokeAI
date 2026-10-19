import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useWidgetSettingsTarget } from '@workbench/settings/useWidgetSettingsTarget';
import { useCallback } from 'react';

import { CANVAS_SETTINGS, readCanvasBooleanSetting } from './canvasSettings';

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const definition = CANVAS_SETTINGS.find((setting) => setting.key === field.id);
  const { disabled, patch, value } = useWidgetSettingsTarget('canvas', target, (values) =>
    definition ? readCanvasBooleanSetting(values, definition) : false
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (definition && typeof next === 'boolean') {
        patch({ [definition.key]: next });
      }
    },
    [definition, patch]
  );
  return (
    <SettingControl
      isModified={definition !== undefined && value !== definition.defaultValue}
      disabled={disabled}
      field={field}
      surface={surface}
      value={value}
      onChange={onChange}
    />
  );
};
