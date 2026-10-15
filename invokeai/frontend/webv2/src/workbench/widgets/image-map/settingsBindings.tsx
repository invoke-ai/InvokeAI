import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { SettingControl } from '@platform/ui/settings/SettingControl';
import { getImageMapClickSelectsCluster, getImageMapShowClusterLabels } from '@workbench/image-map/imageMapSettings';
import { useWidgetSettingsTarget } from '@workbench/settings/useWidgetSettingsTarget';
import { ClusterStrengthField } from '@workbench/widgets/image-map/ClusterStrengthField';
import { useCallback } from 'react';

export const Field = (props: SettingFieldProps) => {
  // Its empty state means "derive it", which no shared control expresses.
  if (props.field.id === 'clusterEps') {
    return <ClusterStrengthField {...props} />;
  }

  return <BooleanField {...props} />;
};

const BooleanField = ({ field, surface, target }: SettingFieldProps) => {
  const { disabled, patch, value } = useWidgetSettingsTarget(
    'image-map',
    target,
    field.id === 'clickSelectsCluster' ? getImageMapClickSelectsCluster : getImageMapShowClusterLabels
  );
  const onChange = useCallback(
    (next: boolean | string | number) => {
      if (typeof next === 'boolean' && (field.id === 'clickSelectsCluster' || field.id === 'showClusterLabels')) {
        patch({ [field.id]: next });
      }
    },
    [field.id, patch]
  );
  const defaultValue =
    field.id === 'clickSelectsCluster' ? getImageMapClickSelectsCluster({}) : getImageMapShowClusterLabels({});
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
