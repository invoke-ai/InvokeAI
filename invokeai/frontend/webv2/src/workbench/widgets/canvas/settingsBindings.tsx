import type { SettingFieldProps, SettingLayer } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useCallback, useMemo } from 'react';

import { CANVAS_SETTINGS, CANVAS_TRACKPAD_NAVIGATION_KEY, readCanvasBooleanSetting } from './canvasSettings';
import { recordTrackpadNavigationChoice } from './trackpadNavigation';

/** Every way the user changes the setting from a row, the layer menu's actions included, counts as their choice. */
const recordingChoice = (layer: SettingLayer | undefined): SettingLayer | undefined => {
  if (layer?.state === 'pinned') {
    return {
      ...layer,
      onFollowDefault: () => {
        recordTrackpadNavigationChoice();
        layer.onFollowDefault();
      },
      onUseAsDefault: () => {
        recordTrackpadNavigationChoice();
        layer.onUseAsDefault();
      },
    };
  }
  const reset = layer?.onResetToBuiltIn;
  return layer && reset
    ? {
        ...layer,
        onResetToBuiltIn: () => {
          recordTrackpadNavigationChoice();
          reset();
        },
      }
    : layer;
};

export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const definition = CANVAS_SETTINGS.find((setting) => setting.key === field.id);
  const read = useCallback(
    (values: Readonly<Record<string, unknown>>) => (definition ? readCanvasBooleanSetting(values, definition) : false),
    [definition]
  );
  const {
    disabled,
    layer: layeredLayer,
    set,
    value,
  } = useLayeredSetting('canvas', field.id, target, read, definition?.defaultValue ?? false, field);
  const isTrackpadNavigation = definition?.key === CANVAS_TRACKPAD_NAVIGATION_KEY;
  const layer = useMemo(
    () => (isTrackpadNavigation ? recordingChoice(layeredLayer) : layeredLayer),
    [isTrackpadNavigation, layeredLayer]
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (definition && typeof next === 'boolean') {
        if (isTrackpadNavigation) {
          recordTrackpadNavigationChoice();
        }
        set(next);
      }
    },
    [definition, isTrackpadNavigation, set]
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
