import { Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { ControlAdapterActionsMenu } from 'features/controlLayers/components/ControlAdapter/ControlAdapterActionsMenu';
import { ControlAdapterOpacityAndFilter } from 'features/controlLayers/components/ControlAdapter/ControlAdapterOpacityAndFilter';
import { ControlAdapterSettings } from 'features/controlLayers/components/ControlAdapter/ControlAdapterSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const ControlAdapter = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'control_adapter' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={onToggle}>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <ControlAdapterOpacityAndFilter />
          <ControlAdapterActionsMenu />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
        {isOpen && <ControlAdapterSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

ControlAdapter.displayName = 'ControlAdapter';
