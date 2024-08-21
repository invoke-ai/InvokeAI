import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { EntityMaskAdapterProviderGate } from 'features/controlLayers/hooks/useEntityMaskAdapter';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

import { InpaintMaskMaskFillColorPicker } from './InpaintMaskMaskFillColorPicker';

type Props = {
  id: string;
};

export const InpaintMask = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'inpaint_mask' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <EntityMaskAdapterProviderGate>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityEnabledToggle />
            <CanvasEntityEditableTitle />
            <Spacer />
            <InpaintMaskMaskFillColorPicker />
            <CanvasEntityDeleteButton />
          </CanvasEntityHeader>
        </CanvasEntityContainer>
      </EntityMaskAdapterProviderGate>
    </EntityIdentifierContext.Provider>
  );
});

InpaintMask.displayName = 'InpaintMask';
