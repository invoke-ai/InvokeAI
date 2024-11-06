import { ControlLayerControlAdapter } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapter';
import { ControlLayerSettingsEmptyState } from 'features/controlLayers/components/ControlLayer/ControlLayerSettingsEmptyState';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsEmpty } from 'features/controlLayers/hooks/useEntityIsEmpty';
import { memo } from 'react';

export const ControlLayerSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const isEmpty = useEntityIsEmpty(entityIdentifier);

  if (isEmpty) {
    return <ControlLayerSettingsEmptyState />;
  }

  return <ControlLayerControlAdapter />;
});

ControlLayerSettings.displayName = 'ControlLayerSettings';
