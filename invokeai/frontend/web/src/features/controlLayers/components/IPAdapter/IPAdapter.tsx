import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { IPAdapterSettings } from 'features/controlLayers/components/IPAdapter/IPAdapterSettings';
import { RefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { memo } from 'react';

type Props = {
  id: string;
};

export const IPAdapter = memo(({ id }: Props) => {
  return (
    <RefImageIdContext.Provider value={id}>
      <CanvasEntityContainer>
        <CanvasEntityHeader ps={4} py={5}>
          <CanvasEntityEditableTitle />
          <Spacer />
          <CanvasEntityHeaderCommonActions />
        </CanvasEntityHeader>
        <IPAdapterSettings />
      </CanvasEntityContainer>
    </RefImageIdContext.Provider>
  );
});

IPAdapter.displayName = 'IPAdapter';
