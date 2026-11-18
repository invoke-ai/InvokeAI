import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useCallback } from 'react';

import { CANVAS_SETTINGS, readCanvasBooleanSetting } from './canvasSettings';

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const definition = CANVAS_SETTINGS.find((setting) => setting.key === field.id);
  const read = useCallback(
    (values: Readonly<Record<string, unknown>>) => (definition ? readCanvasBooleanSetting(values, definition) : false),
    [definition]
  );
  const { disabled, layer, set, value } = useLayeredSetting(
    'canvas',
    field.id,
    target,
    read,
    definition?.defaultValue ?? false,
    field
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (definition && typeof next === 'boolean') {
        set(next);
      }
    },
    [definition, set]
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
