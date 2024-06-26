import { useTranslation } from 'react-i18next';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';

type Props = {
  board_id: string;
  isArchived: boolean;
};

export const BoardTotalsTooltip = ({ board_id, isArchived }: Props) => {
  const { t } = useTranslation();
  const { imagesTotal } = useGetBoardImagesTotalQuery(board_id, {
    selectFromResult: ({ data }) => {
      return { imagesTotal: data?.total ?? 0 };
    },
  });
  const { assetsTotal } = useGetBoardAssetsTotalQuery(board_id, {
    selectFromResult: ({ data }) => {
      return { assetsTotal: data?.total ?? 0 };
    },
  });
  return `${t('boards.imagesWithCount', { count: imagesTotal })}, ${t('boards.assetsWithCount', { count: assetsTotal })}${isArchived ? ` (${t('boards.archived')})` : ''}`;
};
