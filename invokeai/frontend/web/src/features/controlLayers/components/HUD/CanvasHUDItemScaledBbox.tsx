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

  // Fetch system stats with polling every 1 seconds
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

      {/* For Testing Always display system stats (CPU, RAM, GPU) */}
      {systemStats && (
        <>
          <CanvasHUDItem label={t('controlLayers.HUD.cpuUsage')} value={`${systemStats.cpu_usage}%`} />
          <CanvasHUDItem label={t('controlLayers.HUD.ramUsage')} value={`${systemStats.ram_usage}%`} />

          {systemStats.gpu_usage?.map((gpu) => (
            <Fragment key={gpu.id}>
              <CanvasHUDItem label={t('controlLayers.HUD.gpuUsage')} value={`${gpu.load.toFixed(2)}%`} />
              <CanvasHUDItem label={t('controlLayers.HUD.gpuRamUsage')} value={`${gpu.memory} MB`} />
            </Fragment>
          ))}
        </>
      )}
    </>
  );
});

CanvasHUDItemScaledBbox.displayName = 'CanvasHUDItemScaledBbox';
