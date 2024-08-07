import { Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { IPAdapterSettings } from 'features/controlLayers/components/IPAdapter/IPAdapterSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const IPAdapter = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'ip_adapter' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={onToggle}>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
        {isOpen && <IPAdapterSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

IPAdapter.displayName = 'IPAdapter';
