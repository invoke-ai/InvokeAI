import { Grid } from '@invoke-ai/ui-library';
import { CanvasHUDItemAutoSave } from 'features/controlLayers/components/HUD/CanvasHUDItemAutoSave';
import { CanvasHUDItemBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemBbox';
import { CanvasHUDItemGridSize } from 'features/controlLayers/components/HUD/CanvasHUDItemGridSize';
import { CanvasHUDItemScaledBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemScaledBbox';
import { memo } from 'react';

export const CanvasHUD = memo(() => {
  return (
    <Grid
      bg="base.900"
      borderBottomEndRadius="base"
      p={2}
      gap={2}
      borderRadius="base"
      templateColumns="1fr 1fr"
      opacity={0.6}
      minW={64}
    >
      <CanvasHUDItemBbox />
      <CanvasHUDItemScaledBbox />
      <CanvasHUDItemGridSize />
      <CanvasHUDItemAutoSave />
    </Grid>
  );
});

CanvasHUD.displayName = 'CanvasHUD';
