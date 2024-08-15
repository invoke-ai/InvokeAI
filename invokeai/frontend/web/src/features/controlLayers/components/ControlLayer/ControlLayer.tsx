import { Spacer } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { CanvasEntityTitleEdit } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { ControlLayerControlAdapter } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapter';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const ControlLayer = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'control_layer' }), [id]);
  const editing = useBoolean(false);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={editing.setTrue}>
          <CanvasEntityEnabledToggle />
          {editing.isTrue ? <CanvasEntityTitleEdit onStopEditing={editing.setFalse} /> : <CanvasEntityTitle />}
          <Spacer />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
        <CanvasEntitySettingsWrapper>
          <ControlLayerControlAdapter />
        </CanvasEntitySettingsWrapper>
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

ControlLayer.displayName = 'ControlLayer';
