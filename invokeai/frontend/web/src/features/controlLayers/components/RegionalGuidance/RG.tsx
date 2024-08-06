import { useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { RGHeader } from 'features/controlLayers/components/RegionalGuidance/RGHeader';
import { RGSettings } from 'features/controlLayers/components/RegionalGuidance/RGSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const RG = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'regional_guidance' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <RGHeader onToggleVisibility={onToggle} />
        {isOpen && <RGSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

RG.displayName = 'RG';
