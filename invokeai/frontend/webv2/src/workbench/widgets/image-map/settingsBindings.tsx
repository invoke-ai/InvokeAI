import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { getImageMapClickSelectsCluster, getImageMapShowClusterLabels } from '@workbench/image-map/imageMapSettings';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { ClusterStrengthField } from '@workbench/widgets/image-map/ClusterStrengthField';
import { useCallback } from 'react';

const EMPTY = {};

export const Field = (props: SettingFieldProps) => {
  // Its empty state means "derive it", which no shared control expresses.
  if (props.field.id === 'clusterEps') {
    return <ClusterStrengthField {...props} />;
  }

  return <BooleanField {...props} />;
};

const BooleanField = ({ field, surface, target }: SettingFieldProps) => {
  const read = field.id === 'clickSelectsCluster' ? getImageMapClickSelectsCluster : getImageMapShowClusterLabels;
  const { disabled, layer, set, value } = useLayeredSetting('image-map', field.id, target, read, read(EMPTY), field);
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (typeof next === 'boolean' && (field.id === 'clickSelectsCluster' || field.id === 'showClusterLabels')) {
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
