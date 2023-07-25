import { api } from '..';

export const boardImagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Board Images Queries
     */
    // listBoardImages: build.query<
    //   OffsetPaginatedResults_ImageDTO_,
    //   ListBoardImagesArg
    // >({
    //   query: ({ board_id, offset, limit }) => ({
    //     url: `board_images/${board_id}`,
    //     method: 'GET',
    //   }),
    //   providesTags: (result, error, arg) => {
    //     // any list of boardimages
    //     const tags: ApiFullTagDescription[] = [
    //       { type: 'BoardImage', id: `${arg.board_id}_${LIST_TAG}` },
    //     ];
    //     if (result) {
    //       // and individual tags for each boardimage
    //       tags.push(
    //         ...result.items.map(({ board_id, image_name }) => ({
    //           type: 'BoardImage' as const,
    //           id: `${board_id}_${image_name}`,
    //         }))
    //       );
    //     }
    //     return tags;
    //   },
    // }),
  }),
});

// export const { useListBoardImagesQuery } = boardImagesApi;
