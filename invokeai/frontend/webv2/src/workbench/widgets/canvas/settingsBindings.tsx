import type { ResolvedValues } from '@platform/core/resolvedValues';
import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { readBuiltIn, useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useCallback } from 'react';

import {
  CANVAS_SCROLL_INPUT_KEY,
  CANVAS_SETTINGS,
  isCanvasScrollInput,
  readCanvasBooleanSetting,
  readCanvasScrollInput,
} from './canvasSettings';

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const isScrollInput = field.id === CANVAS_SCROLL_INPUT_KEY;
  const definition = CANVAS_SETTINGS.find((setting) => setting.key === field.id);
  const read = useCallback(
    (values: ResolvedValues): boolean | string =>
      isScrollInput ? readCanvasScrollInput(values) : definition ? readCanvasBooleanSetting(values, definition) : false,
    [definition, isScrollInput]
  );
  const { disabled, layer, set, value } = useLayeredSetting<boolean | string>(
    'canvas',
    field.id,
    target,
    read,
    readBuiltIn(read),
    field
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (isScrollInput) {
        if (isCanvasScrollInput(next)) {
          set(next);
        }
      } else if (definition && typeof next === 'boolean') {
        set(next);
      }
    },
    [definition, isScrollInput, set]
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
