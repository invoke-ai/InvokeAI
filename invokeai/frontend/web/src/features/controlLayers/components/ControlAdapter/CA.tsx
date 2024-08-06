import { useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CAHeader } from 'features/controlLayers/components/ControlAdapter/CAEntityHeader';
import { CASettings } from 'features/controlLayers/components/ControlAdapter/CASettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const CA = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'control_adapter' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CAHeader onToggleVisibility={onToggle} />
        {isOpen && <CASettings id={id} />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

CA.displayName = 'CA';
