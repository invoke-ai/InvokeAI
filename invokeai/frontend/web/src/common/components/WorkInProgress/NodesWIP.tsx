import { useTranslation } from 'react-i18next';

export default function NodesWIP() {
  const { t } = useTranslation();
  return (
    <div className="work-in-progress nodes-work-in-progress">
      <h1>{t('common.nodes')}</h1>
      <p>{t('common.nodesDesc')}</p>
    </div>
  );
}
