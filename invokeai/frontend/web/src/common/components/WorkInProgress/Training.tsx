import { useTranslation } from 'react-i18next';

export default function TrainingWIP() {
  const { t } = useTranslation();
  return (
    <div className="work-in-progress nodes-work-in-progress">
      <h1>{t('common.training')}</h1>
      <p>
        {t('common.trainingDesc1')}
        <br />
        <br />
        {t('common.trainingDesc2')}
      </p>
    </div>
  );
}
