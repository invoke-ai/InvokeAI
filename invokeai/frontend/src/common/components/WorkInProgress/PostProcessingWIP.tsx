import { useTranslation } from 'react-i18next';

export const PostProcessingWIP = () => {
  const { t } = useTranslation();
  return (
    <div className="work-in-progress post-processing-work-in-progress">
      <h1>{t('common.postProcessing')}</h1>
      <p>{t('common.postProcessDesc1')}</p>
      <p>{t('common.postProcessDesc2')}</p>
      <p>{t('common.postProcessDesc3')}</p>
    </div>
  );
};
