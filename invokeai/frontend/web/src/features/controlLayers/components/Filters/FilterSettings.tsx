import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { FilterCannyEdgeDetection } from 'features/controlLayers/components/Filters/FilterCannyEdgeDetection';
import { FilterColorMap } from 'features/controlLayers/components/Filters/FilterColorMap';
import { FilterContentShuffle } from 'features/controlLayers/components/Filters/FilterContentShuffle';
import { FilterDepthAnythingDepthEstimation } from 'features/controlLayers/components/Filters/FilterDepthAnythingDepthEstimation';
import { FilterDWOpenposeDetection } from 'features/controlLayers/components/Filters/FilterDWOpenposeDetection';
import { FilterHEDEdgeDetection } from 'features/controlLayers/components/Filters/FilterHEDEdgeDetection';
import { FilterLineartEdgeDetection } from 'features/controlLayers/components/Filters/FilterLineartEdgeDetection';
import { FilterMediaPipeFaceDetection } from 'features/controlLayers/components/Filters/FilterMediaPipeFaceDetection';
import { FilterMLSDDetection } from 'features/controlLayers/components/Filters/FilterMLSDDetection';
import { FilterPiDiNetEdgeDetection } from 'features/controlLayers/components/Filters/FilterPiDiNetEdgeDetection';
import { FilterSpandrel } from 'features/controlLayers/components/Filters/FilterSpandrel';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = { filterConfig: FilterConfig; onChange: (filterConfig: FilterConfig) => void };

export const FilterSettings = memo(({ filterConfig, onChange }: Props) => {
  const { t } = useTranslation();

  if (filterConfig.type === 'canny_edge_detection') {
    return <FilterCannyEdgeDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'color_map') {
    return <FilterColorMap config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'content_shuffle') {
    return <FilterContentShuffle config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'depth_anything_depth_estimation') {
    return <FilterDepthAnythingDepthEstimation config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'dw_openpose_detection') {
    return <FilterDWOpenposeDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'hed_edge_detection') {
    return <FilterHEDEdgeDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'lineart_edge_detection') {
    return <FilterLineartEdgeDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'mediapipe_face_detection') {
    return <FilterMediaPipeFaceDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'mlsd_detection') {
    return <FilterMLSDDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'pidi_edge_detection') {
    return <FilterPiDiNetEdgeDetection config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'spandrel_filter') {
    return <FilterSpandrel config={filterConfig} onChange={onChange} />;
  }

  return (
    <IAINoContentFallback
      py={4}
      label={`${t(`controlLayers.filter.${filterConfig.type}.label`)} has no settings`}
      icon={null}
    />
  );
});

FilterSettings.displayName = 'Filter';
