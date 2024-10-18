import { Divider, Grid } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItemBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemBbox';
import { CanvasHUDItemScaledBbox } from 'features/controlLayers/components/HUD/CanvasHUDItemScaledBbox';
import { CanvasHUDItemStats } from 'features/controlLayers/components/HUD/CanvasHUDItemStats';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';

const selectCanvasSettings = createSelector(selectCanvasSettingsSlice, (canvasSettings) => ({
  showSystemStats: canvasSettings.showSystemStats,
}));

export const CanvasHUD = memo(() => {
  const { showSystemStats } = useAppSelector(selectCanvasSettings);

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

      {showSystemStats && (
        <>
          <Divider gridColumn="span 2" />
          <CanvasHUDItemStats />
        </>
      )}
    </Grid>
  );
});

CanvasHUD.displayName = 'CanvasHUD';
