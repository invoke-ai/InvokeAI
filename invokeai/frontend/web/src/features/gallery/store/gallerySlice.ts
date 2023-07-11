import type { PayloadAction } from '@reduxjs/toolkit';
import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { dateComparator } from 'common/util/dateComparator';
import { filter, forEach, uniq } from 'lodash-es';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { boardsApi } from 'services/api/endpoints/boards';
import { imageDeleted, imagesLoaded } from 'services/api/thunks/image';
import { ImageCategory, ImageDTO } from 'services/api/types';

export const galleryImagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.updated_at, a.updated_at),
});

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = [
  'control',
  'mask',
  'user',
  'other',
];

export const INITIAL_IMAGE_LIMIT = 100;
export const IMAGE_LIMIT = 20;

type RequestState = 'pending' | 'fulfilled' | 'rejected';
type GalleryView = 'images' | 'assets';

// dirty hack to get autocompletion while still accepting any string
type BoardPath =
  | 'all.images'
  | 'all.assets'
  | 'none.images'
  | 'none.assets'
  | 'batch.images'
  | 'batch.assets'
  | `${string}.${GalleryView}`;

const systemBoards = [
  'all.images',
  'all.assets',
  'none.images',
  'none.assets',
  'batch.images',
  'batch.assets',
];

type Boards = Record<
  BoardPath,
  {
    path: BoardPath;
    id: 'all' | 'none' | 'batch' | (string & Record<never, never>);
    view: GalleryView;
    imageNames: string[];
    total: number;
    status: RequestState | undefined;
  }
>;

type AdditionalGalleryState = {
  offset: number;
  limit: number;
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  categories: ImageCategory[];
  selection: string[];
  shouldAutoSwitch: boolean;
  galleryImageMinimumWidth: number;
  isInitialized: boolean;
  galleryView: GalleryView;
  selectedBoardId: 'all' | 'none' | 'batch' | (string & Record<never, never>);
  boards: Boards;
};

const initialBoardState = { imageNames: [], total: 0, status: undefined };

const initialBoards: Boards = {
  'all.images': {
    path: 'all.images',
    id: 'all',
    view: 'images',
    ...initialBoardState,
  },
  'all.assets': {
    path: 'all.assets',
    id: 'all',
    view: 'assets',
    ...initialBoardState,
  },
  'none.images': {
    path: 'none.images',
    id: 'none',
    view: 'images',
    ...initialBoardState,
  },
  'none.assets': {
    path: 'none.assets',
    id: 'none',
    view: 'assets',
    ...initialBoardState,
  },
  'batch.images': {
    path: 'batch.images',
    id: 'batch',
    view: 'images',
    ...initialBoardState,
  },
  'batch.assets': {
    path: 'batch.assets',
    id: 'batch',
    view: 'assets',
    ...initialBoardState,
  },
};

export const initialGalleryState =
  galleryImagesAdapter.getInitialState<AdditionalGalleryState>({
    offset: 0,
    limit: 0,
    total: 0,
    isLoading: true,
    isFetching: true,
    categories: IMAGE_CATEGORIES,
    selection: [],
    shouldAutoSwitch: true,
    galleryImageMinimumWidth: 96,
    galleryView: 'images',
    isInitialized: false,
    selectedBoardId: 'all',
    boards: initialBoards,
  });

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageRemoved: (state, action: PayloadAction<string>) => {
      galleryImagesAdapter.removeOne(state, action.payload);
    },
    imagesRemoved: (state, action: PayloadAction<string[]>) => {
      galleryImagesAdapter.removeMany(state, action.payload);
    },
    imageRangeEndSelected: (state, action: PayloadAction<string>) => {
      const rangeEndImageName = action.payload;
      const lastSelectedImage = state.selection[state.selection.length - 1];

      // get image names for the current board and view
      const imageNames =
        state.boards[`${state.selectedBoardId}.${state.galleryView}`]
          .imageNames;

      // get the index of the last selected image
      const lastClickedIndex = imageNames.findIndex(
        (n) => n === lastSelectedImage
      );

      // get the index of the just-clicked image
      const currentClickedIndex = imageNames.findIndex(
        (n) => n === rangeEndImageName
      );

      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range, selected it!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);

        const imagesToSelect = imageNames.slice(start, end + 1);

        state.selection = uniq(state.selection.concat(imagesToSelect));
      }
    },
    imageSelectionToggled: (state, action: PayloadAction<string>) => {
      if (
        state.selection.includes(action.payload) &&
        state.selection.length > 1
      ) {
        state.selection = state.selection.filter(
          (imageName) => imageName !== action.payload
        );
        return;
      }

      state.selection = uniq(state.selection.concat(action.payload));
    },
    imageSelected: (state, action: PayloadAction<string | null>) => {
      state.selection = action.payload
        ? [action.payload]
        : [String(state.ids[0])];
    },
    shouldAutoSwitchChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoSwitch = action.payload;
    },
    setGalleryImageMinimumWidth: (state, action: PayloadAction<number>) => {
      state.galleryImageMinimumWidth = action.payload;
    },
    setGalleryView: (state, action: PayloadAction<GalleryView>) => {
      state.galleryView = action.payload;
    },
    boardIdSelected: (state, action: PayloadAction<BoardPath>) => {
      const boardId = action.payload;

      if (state.selectedBoardId === boardId) {
        // selected same board, no-op
        return;
      }

      state.selectedBoardId = boardId;

      // handle selecting an unitialized board
      const boardImagesId: BoardPath = `${boardId}.images`;
      const boardAssetsId: BoardPath = `${boardId}.assets`;

      if (!state.boards[boardImagesId]) {
        state.boards[boardImagesId] = {
          path: boardImagesId,
          id: boardId,
          view: 'images',
          ...initialBoardState,
        };
      }

      if (!state.boards[boardAssetsId]) {
        state.boards[boardAssetsId] = {
          path: boardAssetsId,
          id: boardId,
          view: 'assets',
          ...initialBoardState,
        };
      }

      // set the first image as selected
      const firstImageName =
        state.boards[`${boardId}.${state.galleryView}`].imageNames[0];

      state.selection = firstImageName ? [firstImageName] : [];
    },
    isLoadingChanged: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
  extraReducers: (builder) => {
    /**
     * Image deleted
     */
    builder.addCase(imageDeleted.pending, (state, action) => {
      // optimistic update, but no undo :/
      const { image_name } = action.meta.arg;
      // remove image from all boards
      forEach(state.boards, (board) => {
        board.imageNames = board.imageNames.filter((n) => n !== image_name);
      });
      // and selection
      state.selection = state.selection.filter((n) => n !== image_name);
    });
    /**
     * Images loaded into gallery - PENDING
     */
    builder.addCase(imagesLoaded.pending, (state, action) => {
      const { board_id, view } = action.meta.arg;
      state.boards[`${board_id}.${view}`].status = 'pending';
    });
    /**
     * Images loaded into gallery - FULFILLED
     */
    builder.addCase(imagesLoaded.fulfilled, (state, action) => {
      const { items, total } = action.payload;
      const { board_id, view } = action.meta.arg;
      const board = state.boards[`${board_id}.${view}`];

      board.status = 'fulfilled';

      board.imageNames = uniq(
        board.imageNames.concat(items.map((i) => i.image_name))
      );

      board.total = total;

      if (state.selection.length === 0 && items.length) {
        state.selection = [items[0].image_name];
      }
    });
    /**
     * Images loaded into gallery - REJECTED
     */
    builder.addCase(imagesLoaded.rejected, (state, action) => {
      const { board_id, view } = action.meta.arg;
      state.boards[`${board_id}.${view}`].status = 'rejected';
    });
    /**
     * Image added to board
     */
    builder.addMatcher(
      boardImagesApi.endpoints.addBoardImage.matchFulfilled,
      (state, action) => {
        const { board_id, image_name } = action.meta.arg.originalArgs;
        // update user board stores
        const userBoards = selectUserBoards(state);
        userBoards.forEach((board) => {
          // only update the current view
          if (board.view !== state.galleryView) {
            return;
          }

          if (board_id === board.id) {
            // add image to the board
            board.imageNames = uniq(board.imageNames.concat(image_name));
          } else {
            // remove image from other boards
            board.imageNames = board.imageNames.filter((n) => n !== image_name);
          }
        });
      }
    );
    /**
     * Many images added to board
     */
    builder.addMatcher(
      boardImagesApi.endpoints.addManyBoardImages.matchFulfilled,
      (state, action) => {
        const { board_id, image_names } = action.meta.arg.originalArgs;
        // update local board stores
        forEach(state.boards, (board, board_id) => {
          // only update the current view
          if (board_id === board.id) {
            // add images to the board
            board.imageNames = uniq(board.imageNames.concat(image_names));
          } else {
            // remove images from other boards
            board.imageNames = board.imageNames.filter((n) =>
              image_names.includes(n)
            );
          }
        });
      }
    );
    /**
     * Board deleted (not images)
     */
    builder.addMatcher(
      boardsApi.endpoints.deleteBoard.matchFulfilled,
      (state, action) => {
        const deletedBoardId = action.meta.arg.originalArgs;
        if (deletedBoardId === state.selectedBoardId) {
          state.selectedBoardId = 'all';
        }
        // remove board from local store
        delete state.boards[`${deletedBoardId}.images`];
        delete state.boards[`${deletedBoardId}.assets`];
      }
    );
    /**
     * Board deleted (with images)
     */
    builder.addMatcher(
      boardsApi.endpoints.deleteBoardAndImages.matchFulfilled,
      (state, action) => {
        const { deleted_images } = action.payload;
        const deletedBoardId = action.meta.arg.originalArgs;
        // remove images from all boards
        forEach(state.boards, (board) => {
          // remove images from all boards
          board.imageNames = board.imageNames.filter((n) =>
            deleted_images.includes(n)
          );
        });

        delete state.boards[`${deletedBoardId}.images`];
        delete state.boards[`${deletedBoardId}.assets`];
      }
    );
    /**
     * Image removed from board; i.e. Board reset for image
     */
    builder.addMatcher(
      boardImagesApi.endpoints.deleteBoardImage.matchFulfilled,
      (state, action) => {
        const { image_name } = action.meta.arg.originalArgs;
        // remove from all user boards (skip all, none, batch)
        const userBoards = selectUserBoards(state);
        userBoards.forEach((board) => {
          board.imageNames = board.imageNames.filter((n) => n !== image_name);
        });
      }
    );
    /**
     * Many images removed from board; i.e. Board reset for many images
     */
    builder.addMatcher(
      boardImagesApi.endpoints.deleteManyBoardImages.matchFulfilled,
      (state, action) => {
        const { image_names } = action.meta.arg.originalArgs;
        // remove images from all boards
        forEach(state.imageNamesByIdAndView, (board) => {
          // only update the current view
          const view = board[state.galleryView];
          view.imageNames = view.imageNames.filter((n) =>
            image_names.includes(n)
          );
        });
      }
    );
  },
});

export const {
  selectAll: selectImagesAll,
  selectById: selectImagesById,
  selectEntities: selectImagesEntities,
  selectIds: selectImagesIds,
  selectTotal: selectImagesTotal,
} = galleryImagesAdapter.getSelectors<RootState>((state) => state.gallery);

export const {
  imagesRemoved,
  imageRangeEndSelected,
  imageSelectionToggled,
  imageSelected,
  shouldAutoSwitchChanged,
  setGalleryImageMinimumWidth,
  setGalleryView,
  boardIdSelected,
  isLoadingChanged,
} = gallerySlice.actions;

export default gallerySlice.reducer;

const selectUserBoards = (state: typeof initialGalleryState) =>
  filter(state.boards, (board, path) => !systemBoards.includes(path));

const selectCurrentBoard = (state: typeof initialGalleryState) =>
  state.boards[`${state.selectedBoardId}.${state.galleryView}`];

const isImagesView = (board: BoardPath) => board.split('.')[1] === 'images';

const isAssetsView = (board: BoardPath) => board.split('.')[1] === 'assets';
