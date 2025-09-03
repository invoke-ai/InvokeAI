import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO | null;
  boardCounts: {
    image_count: number;
    asset_count: number;
    video_count: number;
  };
};

export const BoardTooltip = ({ board, boardCounts }: Props) => {
  const { t } = useTranslation();
  const crossOrigin = useStore($crossOrigin);

  const isVideoEnabled = useFeatureStatus('video');

  const { currentData: coverImage } = useGetImageDTOQuery(board?.cover_image_name ?? skipToken);

  return (
    <Flex flexDir="column" alignItems="center" gap={1}>
      {coverImage && (
        <Image
          src={coverImage.thumbnail_url}
          crossOrigin={crossOrigin}
          draggable={false}
          objectFit="cover"
          maxW={150}
          aspectRatio="1/1"
          borderRadius="base"
          borderBottomRadius="lg"
        />
      )}
      <Flex flexDir="column" alignItems="center">
        {board && <Text fontWeight="semibold">{board.board_name}</Text>}
        <Text noOfLines={1}>
          {t('boards.imagesWithCount', { count: boardCounts.image_count })},{' '}
          {t('boards.assetsWithCount', { count: boardCounts.asset_count })}
        </Text>
        {isVideoEnabled && <Text noOfLines={1}>{t('boards.videosWithCount', { count: boardCounts.video_count })}</Text>}
        {board?.archived && <Text>({t('boards.archived')})</Text>}
      </Flex>
    </Flex>
  );
};
