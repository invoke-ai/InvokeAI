import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
import { filterConfigChanged } from 'features/controlLayers/store/canvasV2Slice';
import { type FilterConfig, IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const FilterSettings = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const config = useAppSelector((s) => s.canvasV2.filter.config);
  const updateFilter = useCallback(
    (config: FilterConfig) => {
      dispatch(filterConfigChanged({ config }));
    },
    [dispatch]
  );

  if (config.type === 'canny_image_processor') {
    return <FilterCanny config={config} onChange={updateFilter} />;
  }

  if (config.type === 'color_map_image_processor') {
    return <FilterColorMap config={config} onChange={updateFilter} />;
  }

  if (config.type === 'content_shuffle_image_processor') {
    return <FilterContentShuffle config={config} onChange={updateFilter} />;
  }

  if (config.type === 'depth_anything_image_processor') {
    return <FilterDepthAnything config={config} onChange={updateFilter} />;
  }

  if (config.type === 'dw_openpose_image_processor') {
    return <FilterDWOpenpose config={config} onChange={updateFilter} />;
  }

  if (config.type === 'hed_image_processor') {
    return <FilterHed config={config} onChange={updateFilter} />;
  }

  if (config.type === 'lineart_image_processor') {
    return <FilterLineart config={config} onChange={updateFilter} />;
  }

  if (config.type === 'mediapipe_face_processor') {
    return <FilterMediapipeFace config={config} onChange={updateFilter} />;
  }

  if (config.type === 'midas_depth_image_processor') {
    return <FilterMidasDepth config={config} onChange={updateFilter} />;
  }

  if (config.type === 'mlsd_image_processor') {
    return <FilterMlsdImage config={config} onChange={updateFilter} />;
  }

  if (config.type === 'pidi_image_processor') {
    return <FilterPidi config={config} onChange={updateFilter} />;
  }

  return <IAINoContentFallback label={`${t(IMAGE_FILTERS[config.type].labelTKey)} has no settings`} icon={null} />;
});

FilterSettings.displayName = 'Filter';
