import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import {
  BUILT_IN_PROJECT_SETTINGS,
  type LayeredProjectSettingKey,
  PROJECT_SETTINGS_OWNER,
} from '@workbench/settings/projectSettings';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useCallback, useMemo } from 'react';

const readProjectSetting = (key: LayeredProjectSettingKey) => (values: Readonly<Record<string, unknown>>) => {
  const value = values[key];
  return typeof value === 'boolean' ? value : BUILT_IN_PROJECT_SETTINGS[key];
};

/**
 * The project's generation flags behind the Generate widget's settings: this project's pin inside a project, the
 * user's default outside one. They live in the project settings, not the Generate widget's values.
 */
export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const key = field.id as LayeredProjectSettingKey;
  const read = useMemo(() => readProjectSetting(key), [key]);
  const { disabled, layer, set, value } = useLayeredSetting(
    PROJECT_SETTINGS_OWNER,
    key,
    target,
    read,
    BUILT_IN_PROJECT_SETTINGS[key],
    field
  );
  const onChange = useCallback(
    (next: boolean | number | string) => {
      if (typeof next === 'boolean') {
        set(next);
      }
    },
    [set]
  );
  return (
    <SettingControl
      layer={layer}
      field={field}
      surface={surface}
      value={value}
      disabled={disabled}
      onChange={onChange}
    />
  );
};
