import { useTranslation } from 'react-i18next';

type Props = {
  imageCount: number;
  assetCount: number;
  isArchived: boolean;
};

export const BoardTotalsTooltip = ({ imageCount, assetCount, isArchived }: Props) => {
  const { t } = useTranslation();
  return `${t('boards.imagesWithCount', { count: imageCount })}, ${t('boards.assetsWithCount', { count: assetCount })}${isArchived ? ` (${t('boards.archived')})` : ''}`;
};
