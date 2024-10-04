import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { IPAdapterSettings } from 'features/controlLayers/components/IPAdapter/IPAdapterSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const IPAdapter = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'reference_image' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader ps={4} py={5}>
          <CanvasEntityEditableTitle />
          <Spacer />
          <CanvasEntityHeaderCommonActions />
        </CanvasEntityHeader>
        <IPAdapterSettings />
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

IPAdapter.displayName = 'IPAdapter';
