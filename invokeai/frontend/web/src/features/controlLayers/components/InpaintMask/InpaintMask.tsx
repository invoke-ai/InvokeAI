import { Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { InpaintMaskActionsMenu } from 'features/controlLayers/components/InpaintMask/InpaintMaskActionsMenu';
import { InpaintMaskSettings } from 'features/controlLayers/components/InpaintMask/InpaintMaskSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

import { InpaintMaskMaskFillColorPicker } from './InpaintMaskMaskFillColorPicker';

export const InpaintMask = memo(() => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id: 'inpaint_mask', type: 'inpaint_mask' }), []);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });
  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={onToggle}>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <InpaintMaskMaskFillColorPicker />
          <InpaintMaskActionsMenu />
        </CanvasEntityHeader>
        {isOpen && <InpaintMaskSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

InpaintMask.displayName = 'InpaintMask';
