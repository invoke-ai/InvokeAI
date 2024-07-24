import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useTranslation } from 'react-i18next';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO | null;
};

export const BoardTooltip = ({ board }: Props) => {
  const { t } = useTranslation();
  const { imagesTotal } = useGetBoardImagesTotalQuery(board?.board_id || 'none', {
    selectFromResult: ({ data }) => {
      return { imagesTotal: data?.total ?? 0 };
    },
  });
  const { assetsTotal } = useGetBoardAssetsTotalQuery(board?.board_id || 'none', {
    selectFromResult: ({ data }) => {
      return { assetsTotal: data?.total ?? 0 };
    },
  });
  const { currentData: coverImage } = useGetImageDTOQuery(board?.cover_image_name ?? skipToken);

  return (
    <Flex flexDir="column" alignItems="center" gap={1}>
      {coverImage && (
        <Image
          src={coverImage.thumbnail_url}
          draggable={false}
          objectFit="cover"
          maxW={150}
          aspectRatio="1/1"
          borderRadius="base"
          borderBottomRadius="lg"
        />
      )}
      <Flex flexDir="column" alignItems="center">
        <Text noOfLines={1}>
          {t('boards.imagesWithCount', { count: imagesTotal })}, {t('boards.assetsWithCount', { count: assetsTotal })}
        </Text>
        {board?.archived && <Text>({t('boards.archived')})</Text>}
      </Flex>
    </Flex>
  );
};
