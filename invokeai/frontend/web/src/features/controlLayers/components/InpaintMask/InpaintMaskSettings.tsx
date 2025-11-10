import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { InpaintMaskDenoiseLimitSlider } from 'features/controlLayers/components/InpaintMask/InpaintMaskDenoiseLimitSlider';
import { InpaintMaskNoiseSlider } from 'features/controlLayers/components/InpaintMask/InpaintMaskNoiseSlider';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectActiveCanvas, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

const buildSelectHasDenoiseLimit = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) =>
  createSelector(selectActiveCanvas, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskSettings');
    return entity.denoiseLimit !== undefined;
  });

const buildSelectHasNoiseLevel = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) =>
  createSelector(selectActiveCanvas, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskSettings');
    return entity.noiseLevel !== undefined;
  });

export const InpaintMaskSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const selectHasDenoiseLimit = useMemo(() => buildSelectHasDenoiseLimit(entityIdentifier), [entityIdentifier]);
  const selectHasNoiseLevel = useMemo(() => buildSelectHasNoiseLevel(entityIdentifier), [entityIdentifier]);

  const hasDenoiseLimit = useAppSelector(selectHasDenoiseLimit);
  const hasNoiseLevel = useAppSelector(selectHasNoiseLevel);

  if (!hasNoiseLevel && !hasDenoiseLimit) {
    // If we show the <InpaintMaskAddButtons /> below, we can remove this check.
    // Until then, if there are no sliders to show for the mask settings, return null. This prevents rendering an
    // empty settings wrapper div, which adds unnecessary space in the UI.
    return null;
  }

  return (
    <CanvasEntitySettingsWrapper>
      {/* {!hasNoiseLevel && !hasDenoiseLimit && <InpaintMaskAddButtons />} */}
      {hasNoiseLevel && <InpaintMaskNoiseSlider />}
      {hasDenoiseLimit && <InpaintMaskDenoiseLimitSlider />}
    </CanvasEntitySettingsWrapper>
  );
});

InpaintMaskSettings.displayName = 'InpaintMaskSettings';
