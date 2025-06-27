import { useAppSelector } from 'app/store/storeHooks';
import { selectNonRasterLayersIsHidden } from 'features/controlLayers/store/selectors';

export const useNonRasterLayersIsHidden = (): boolean => {
  return useAppSelector(selectNonRasterLayersIsHidden);
};
