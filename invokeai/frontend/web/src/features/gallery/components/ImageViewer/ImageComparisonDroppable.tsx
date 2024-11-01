import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { type SetComparisonImageActionData, setComparisonImageActionApi } from 'features/imageActions/actions';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const targetData = useMemo<SetComparisonImageActionData>(() => setComparisonImageActionApi.getData(), []);

  return <DndDropTarget targetData={targetData} label={t('gallery.selectForCompare')} />;
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';
