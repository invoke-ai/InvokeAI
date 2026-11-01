import { useTranslation } from 'react-i18next';

import { QueueMenuItems, useQueueActions } from './queueMenuActions';

/** The top bar's action model plus history clean-up; every entry confirms by the same rules wherever it appears. */
export const QueueHeaderMenu = () => {
  const { t } = useTranslation();
  const actions = useQueueActions();

  return <QueueMenuItems actions={actions} includeClear label={t('widgets.labels.queue')} />;
};
