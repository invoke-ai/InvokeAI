import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectScaledSize, selectScaleMethod } from 'features/controlLayers/store/selectors';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSystemStatsQuery } from 'services/api/endpoints/appInfo';

export const CanvasHUDItemScaledBbox = memo(() => {
  const { t } = useTranslation();
  const scaleMethod = useAppSelector(selectScaleMethod);
  const scaledSize = useAppSelector(selectScaledSize);

  // Fetch system stats with polling every 1 second
  const { data: systemStats } = useGetSystemStatsQuery(undefined, {
    pollingInterval: 1000,
  });

  return (
    <>
      {/* Only display scaled bounding box size if the scale method is not 'none' */}
      {scaleMethod !== 'none' && (
        <CanvasHUDItem
          label={t('controlLayers.HUD.scaledBbox')}
          value={`${scaledSize.width}×${scaledSize.height} px`}
        />
      )}

      {/* Display system stats (CPU, RAM, GPU) with temperatures */}
      {systemStats && (
        <>
          <CanvasHUDItem label={t('controlLayers.HUD.cpuUsage')} value={`${systemStats.cpu_usage.toFixed(0)}%`} />
          <CanvasHUDItem label={t('controlLayers.HUD.ramUsage')} value={`${systemStats.ram_usage.toFixed(0)} MB`} />

          {systemStats.gpu_usage?.map((gpu) => (
            <Fragment key={gpu.id}>
              <CanvasHUDItem label={t('controlLayers.HUD.gpuUsage')} value={`${gpu.load.toFixed(0)}%`} />
              <CanvasHUDItem label={t('controlLayers.HUD.gpuVram')} value={`${gpu.memory} MB`} />
              <CanvasHUDItem
                label={t('controlLayers.HUD.gpuTemp')}
                value={`${gpu.temperature?.toFixed(1) || 'N/A'}°C`}
              />
            </Fragment>
          ))}
        </>
      )}
    </>
  );
});

CanvasHUDItemScaledBbox.displayName = 'CanvasHUDItemScaledBbox';
