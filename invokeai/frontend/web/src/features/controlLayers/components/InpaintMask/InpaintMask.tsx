import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { CanvasEntityStateGate } from 'features/controlLayers/contexts/CanvasEntityStateGate';
import { InpaintMaskAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const InpaintMask = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'inpaint_mask'>>(() => ({ id, type: 'inpaint_mask' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <InpaintMaskAdapterGate>
        <CanvasEntityStateGate entityIdentifier={entityIdentifier}>
          <CanvasEntityContainer>
            <CanvasEntityHeader>
              <CanvasEntityPreviewImage />
              <CanvasEntityEditableTitle />
              <Spacer />
              <CanvasEntityHeaderCommonActions />
            </CanvasEntityHeader>
          </CanvasEntityContainer>
        </CanvasEntityStateGate>
      </InpaintMaskAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

InpaintMask.displayName = 'InpaintMask';
