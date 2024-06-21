import { useTranslation } from 'react-i18next';
import { useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';

type Props = {
  board_id: string;
};

export const BoardTotalsTooltip = ({ board_id }: Props) => {
  const { t } = useTranslation();
  const { imagesTotal } = useGetBoardImagesTotalQuery(board_id, {
    selectFromResult: ({ data }) => {
      return { imagesTotal: data?.total ?? 0 };
    },
  });
  const { assetsTotal } = useGetBoardImagesTotalQuery(board_id, {
    selectFromResult: ({ data }) => {
      return { assetsTotal: data?.total ?? 0 };
    },
  });
  return `${t('boards.imagesWithCount', { count: imagesTotal })}, ${t('boards.assetsWithCount', { count: assetsTotal })}`;
};
