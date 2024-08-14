import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { LayerActionsMenu } from 'features/controlLayers/components/Layer/LayerActionsMenu';
import { LayerSettings } from 'features/controlLayers/components/Layer/LayerSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

import { LayerOpacity } from './LayerOpacity';

type Props = {
  id: string;
};

export const Layer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'layer' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <LayerOpacity />
          <LayerActionsMenu />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
        <LayerSettings />
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

Layer.displayName = 'Layer';
