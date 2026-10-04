import type { SetComparisonImageDndTargetData } from 'features/dnd/dnd';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const dndTargetData = useMemo<SetComparisonImageDndTargetData>(() => setComparisonImageDndTarget.getData(), []);

  return (
    <DndDropTarget
      dndTarget={setComparisonImageDndTarget}
      dndTargetData={dndTargetData}
      label={t('gallery.selectForCompare')}
    />
  );
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';
