import { useAppSelector } from 'app/store/storeHooks';
import { Dnd } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { selectComparisonImages } from './common';

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const comparisonImages = useAppSelector(selectComparisonImages);
  const targetData = useMemo<Dnd.types['TargetDataTypeMap']['selectForCompare']>(() => {
    const { firstImage, secondImage } = comparisonImages;
    return Dnd.Target.selectForCompare.getData({
      firstImageName: firstImage?.image_name,
      secondImageName: secondImage?.image_name,
    });
  }, [comparisonImages]);

  return <DndDropTarget targetData={targetData} label={t('gallery.selectForCompare')} />;
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';
