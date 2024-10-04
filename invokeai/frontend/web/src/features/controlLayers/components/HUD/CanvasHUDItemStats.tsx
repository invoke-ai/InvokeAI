import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSystemStatsQuery } from 'services/api/endpoints/appInfo';

export const CanvasHUDItemStats = memo(() => {
  const { t } = useTranslation();

  // Fetch system stats with polling every 1 second
  const { data: systemStats } = useGetSystemStatsQuery(undefined, {
    pollingInterval: 1000,
  });

  if (!systemStats) {
    return null;
  }

  return (
    <>
      {/* Display system stats (CPU, RAM, GPU) */}
      <CanvasHUDItem label={t('controlLayers.HUD.cpuUsage')} value={`${systemStats.cpu_usage.toFixed(0)}%`} />
      <CanvasHUDItem label={t('controlLayers.HUD.ramUsage')} value={`${systemStats.ram_usage.toFixed(0)} MB`} />

      {systemStats.gpu_usage?.map((gpu) => (
        <Fragment key={gpu.id}>
          <CanvasHUDItem label={t('controlLayers.HUD.gpuUsage')} value={`${gpu.load.toFixed(0)}%`} />
          <CanvasHUDItem label={t('controlLayers.HUD.gpuVram')} value={`${gpu.memory} MB`} />
          {gpu.temperature !== undefined && (
            <CanvasHUDItem label={t('controlLayers.HUD.gpuTemp')} value={`${gpu.temperature} °C`} />
          )}
        </Fragment>
      ))}
    </>
  );
});

CanvasHUDItemStats.displayName = 'CanvasHUDItemStats';
