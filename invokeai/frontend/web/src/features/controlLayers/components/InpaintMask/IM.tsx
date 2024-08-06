import { useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { IMHeader } from 'features/controlLayers/components/InpaintMask/IMHeader';
import { IMSettings } from 'features/controlLayers/components/InpaintMask/IMSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

export const IM = memo(() => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id: 'inpaint_mask', type: 'inpaint_mask' }), []);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });
  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <IMHeader onToggleVisibility={onToggle} />
        {isOpen && <IMSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

IM.displayName = 'IM';
