import { Grid } from '@invoke-ai/ui-library';
import { CanvasHUDItemBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemBbox';
import { CanvasHUDItemScaledBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemScaledBbox';
import { memo } from 'react';

export const CanvasHUD = memo(() => {
  return (
    <Grid
      bg="base.900"
      borderBottomEndRadius="base"
      p={2}
      gap={1}
      borderRadius="base"
      templateColumns="1fr 1fr"
      opacity={0.6}
      minW={64}
    >
      <CanvasHUDItemBbox />
      <CanvasHUDItemScaledBbox />
    </Grid>
  );
});

CanvasHUD.displayName = 'CanvasHUD';
