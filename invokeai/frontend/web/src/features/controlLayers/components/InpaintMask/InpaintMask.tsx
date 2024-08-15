import { Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityGroupTitle } from 'features/controlLayers/components/common/CanvasEntityGroupTitle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { InpaintMaskMaskFillColorPicker } from './InpaintMaskMaskFillColorPicker';

export const InpaintMask = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id: 'inpaint_mask', type: 'inpaint_mask' }), []);
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'inpaint_mask'));

  return (
    <>
      <CanvasEntityGroupTitle title={t('controlLayers.inpaintMask')} isSelected={isSelected} />
      <EntityIdentifierContext.Provider value={entityIdentifier}>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityEnabledToggle />
            <CanvasEntityTitle />
            <Spacer />
            <InpaintMaskMaskFillColorPicker />
          </CanvasEntityHeader>
        </CanvasEntityContainer>
      </EntityIdentifierContext.Provider>
    </>
  );
});

InpaintMask.displayName = 'InpaintMask';
