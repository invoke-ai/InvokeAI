import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

type Props = {
  imageCount: number;
  assetCount: number;
  isArchived: boolean;
  coverImageName?: string | null;
};

export const BoardTooltip = ({ imageCount, assetCount, isArchived, coverImageName }: Props) => {
  const { t } = useTranslation();
  const { currentData: coverImage } = useGetImageDTOQuery(coverImageName ?? skipToken);

  const totalString = useMemo(() => {
    return `${t('boards.imagesWithCount', { count: imageCount })}, ${t('boards.assetsWithCount', { count: assetCount })}${isArchived ? ` (${t('boards.archived')})` : ''}`;
  }, [assetCount, imageCount, isArchived, t]);

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
          mt={1}
        />
      )}
      <Flex flexDir="column" alignItems="center">
        <Text noOfLines={1}>{totalString}</Text>
      </Flex>
    </Flex>
  );
};
