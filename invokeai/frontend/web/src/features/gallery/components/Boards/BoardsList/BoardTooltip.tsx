import { Flex, Icon, Image, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useMediaUrl } from 'features/auth/store/mediaCookieRefresh';
import { useTranslation } from 'react-i18next';
import { PiImageSquare } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO | null;
  boardCounts: {
    image_count: number;
    video_count: number;
    asset_count: number;
  };
};

export const BoardTooltip = ({ board, boardCounts }: Props) => {
  const { t } = useTranslation();

  // Backend picks a single cover — either an image or a video. Prefer the video when set.
  const { currentData: coverVideo } = useGetVideoDTOQuery(board?.cover_video_name ?? skipToken);
  const { currentData: coverImage } = useGetImageDTOQuery(
    board?.cover_video_name ? skipToken : (board?.cover_image_name ?? skipToken)
  );
  const thumbnailUrl = coverVideo?.thumbnail_url ?? coverImage?.thumbnail_url;
  const refreshedThumbnailUrl = useMediaUrl(thumbnailUrl);

  return (
    <Flex flexDir="column" alignItems="center" gap={1}>
      {refreshedThumbnailUrl && (
        <Image
          src={refreshedThumbnailUrl}
          draggable={false}
          objectFit="cover"
          maxW={150}
          aspectRatio="1/1"
          borderRadius="base"
          borderBottomRadius="lg"
          // Thumbnail generation is best-effort on the backend (a video can exist whose
          // thumbnail 404s) — degrade to an icon rather than a broken image.
          fallbackStrategy="onError"
          fallback={
            <Flex w={150} aspectRatio="1/1" justifyContent="center" alignItems="center">
              <Icon boxSize={16} as={PiImageSquare} opacity={0.7} color="base.500" />
            </Flex>
          }
        />
      )}
      <Flex flexDir="column" alignItems="center">
        {board && <Text fontWeight="semibold">{board.board_name}</Text>}
        <Text noOfLines={1}>
          {t('boards.imagesWithCount', { count: boardCounts.image_count })},{' '}
          {t('boards.videosWithCount', { count: boardCounts.video_count })},{' '}
          {t('boards.assetsWithCount', { count: boardCounts.asset_count })}
        </Text>
        {board?.archived && <Text>({t('boards.archived')})</Text>}
      </Flex>
    </Flex>
  );
};
