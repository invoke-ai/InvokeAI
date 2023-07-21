import {
  BoardId,
  INITIAL_IMAGE_LIMIT,
} from 'features/gallery/store/gallerySlice';
import {
  ListImagesArgs,
  useGetBoardAssetsTotalQuery,
  useGetBoardImagesTotalQuery,
} from '../endpoints/images';

const baseQueryArgs: ListImagesArgs = {
  offset: 0,
  limit: INITIAL_IMAGE_LIMIT,
  is_intermediate: false,
};

export const useBoardTotal = (board_id: BoardId) => {
  const { data: totalImages } = useGetBoardImagesTotalQuery(board_id);
  const { data: totalAssets } = useGetBoardAssetsTotalQuery(board_id);
  // const imagesQueryArg = useMemo(() => {
  //   const categories = IMAGE_CATEGORIES;
  //   return { board_id, categories, ...baseQueryArgs };
  // }, [board_id]);

  // const assetsQueryArg = useMemo(() => {
  //   const categories = ASSETS_CATEGORIES;
  //   return { board_id, categories, ...baseQueryArgs };
  // }, [board_id]);

  // const { total: totalImages } = useListImagesQuery(
  //   imagesQueryArg ?? skipToken,
  //   {
  //     selectFromResult: ({ currentData }) => ({ total: currentData?.total }),
  //   }
  // );

  // const { total: totalAssets } = useListImagesQuery(
  //   assetsQueryArg ?? skipToken,
  //   {
  //     selectFromResult: ({ currentData }) => ({ total: currentData?.total }),
  //   }
  // );

  return { totalImages, totalAssets };
};
