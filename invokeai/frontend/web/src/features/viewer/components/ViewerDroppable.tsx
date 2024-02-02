import IAIDroppable from 'common/components/IAIDroppable';
import type { ViewerImageDropData } from 'features/dnd/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const viewerDropData: ViewerImageDropData = {
  id: 'viewer-image',
  actionType: 'SET_VIEWER_IMAGE',
};

export const ViewerDroppable = memo(() => {
  const { t } = useTranslation();

  return <IAIDroppable data={viewerDropData} dropLabel={t('viewer.dropLabel')} />;
});

ViewerDroppable.displayName = 'ViewerDroppable';
