import { useAppStore } from 'app/store/storeHooks';
import { DndDropTarget } from 'features/dnd2/DndDropTarget';
import type { SelectForCompareDndTargetData } from 'features/dnd2/types';
import { selectForCompareDndTarget } from 'features/dnd2/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { selectComparisonImages } from './common';

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const targetData = useMemo<SelectForCompareDndTargetData>(() => {
    const { firstImage, secondImage } = selectComparisonImages(store.getState());
    return selectForCompareDndTarget.getData({
      dndId: 'current-image',
      firstImageName: firstImage?.image_name,
      secondImageName: secondImage?.image_name,
    });
  }, [store]);

  return <DndDropTarget targetData={targetData} label={t('gallery.selectForCompare')} />;
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';
