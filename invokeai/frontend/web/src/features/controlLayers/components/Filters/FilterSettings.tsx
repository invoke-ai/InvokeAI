import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { FilterCanny } from 'features/controlLayers/components/Filters/FilterCanny';
import { FilterColorMap } from 'features/controlLayers/components/Filters/FilterColorMap';
import { FilterContentShuffle } from 'features/controlLayers/components/Filters/FilterContentShuffle';
import { FilterDepthAnything } from 'features/controlLayers/components/Filters/FilterDepthAnything';
import { FilterDWOpenpose } from 'features/controlLayers/components/Filters/FilterDWOpenpose';
import { FilterHed } from 'features/controlLayers/components/Filters/FilterHed';
import { FilterLineart } from 'features/controlLayers/components/Filters/FilterLineart';
import { FilterMediapipeFace } from 'features/controlLayers/components/Filters/FilterMediapipeFace';
import { FilterMidasDepth } from 'features/controlLayers/components/Filters/FilterMidasDepth';
import { FilterMlsdImage } from 'features/controlLayers/components/Filters/FilterMlsdImage';
import { FilterPidi } from 'features/controlLayers/components/Filters/FilterPidi';
import type { FilterConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = { filterConfig: FilterConfig; onChange: (filterConfig: FilterConfig) => void };

export const FilterSettings = memo(({ filterConfig, onChange }: Props) => {
  const { t } = useTranslation();

  if (filterConfig.type === 'canny_image_processor') {
    return <FilterCanny config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'color_map_image_processor') {
    return <FilterColorMap config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'content_shuffle_image_processor') {
    return <FilterContentShuffle config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'depth_anything_image_processor') {
    return <FilterDepthAnything config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'dw_openpose_image_processor') {
    return <FilterDWOpenpose config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'hed_image_processor') {
    return <FilterHed config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'lineart_image_processor') {
    return <FilterLineart config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'mediapipe_face_processor') {
    return <FilterMediapipeFace config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'midas_depth_image_processor') {
    return <FilterMidasDepth config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'mlsd_image_processor') {
    return <FilterMlsdImage config={filterConfig} onChange={onChange} />;
  }

  if (filterConfig.type === 'pidi_image_processor') {
    return <FilterPidi config={filterConfig} onChange={onChange} />;
  }

  return (
    <IAINoContentFallback
      py={4}
      label={`${t(IMAGE_FILTERS[filterConfig.type].labelTKey)} has no settings`}
      icon={null}
    />
  );
});

FilterSettings.displayName = 'Filter';
