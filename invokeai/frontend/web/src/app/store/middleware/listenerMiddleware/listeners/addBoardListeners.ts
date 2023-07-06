import { createAction } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import {
  imageUpdatedMany,
  imageUpdatedOne,
} from 'features/gallery/store/gallerySlice';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardListeners = () => {
  // add image to board - fulfilled
  startAppListening({
    matcher: boardImagesApi.endpoints.addImageToBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Image added to board'
      );

      dispatch(
        imageUpdatedOne({
          id: image_name,
          changes: { board_id },
        })
      );
    },
  });

  // add image to board - rejected
  startAppListening({
    matcher: boardImagesApi.endpoints.addImageToBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Problem adding image to board'
      );
    },
  });

  // remove image from board - fulfilled
  startAppListening({
    matcher: boardImagesApi.endpoints.removeImageFromBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const image_name = action.meta.arg.originalArgs;

      moduleLog.debug({ data: { image_name } }, 'Image removed from board');

      dispatch(
        imageUpdatedOne({
          id: image_name,
          changes: { board_id: undefined },
        })
      );
    },
  });

  // remove image from board - rejected
  startAppListening({
    matcher: boardImagesApi.endpoints.removeImageFromBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const image_name = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { image_name } },
        'Problem removing image from board'
      );
    },
  });

  // gallery selection added to board
  startAppListening({
    actionCreator: gallerySelectionAddedToBoard,
    effect: (action, { getState, dispatch }) => {
      const { board_id } = action.payload;
      const image_names = getState().gallery.selection;
      dispatch(
        boardImagesApi.endpoints.addManyImagesToBoard.initiate({
          board_id,
          image_names,
        })
      );
    },
  });

  // gallery selection removed from board
  startAppListening({
    actionCreator: gallerySelectionRemovedFromBoard,
    effect: (action, { getState, dispatch }) => {
      const image_names = getState().gallery.selection;
      dispatch(
        boardImagesApi.endpoints.removeManyImagesFromBoard.initiate(image_names)
      );
    },
  });

  // batch selection added to board
  startAppListening({
    actionCreator: batchSelectionAddedToBoard,
    effect: (action, { getState, dispatch }) => {
      const { board_id } = action.payload;
      const image_names = getState().batch.selection;
      dispatch(
        boardImagesApi.endpoints.addManyImagesToBoard.initiate({
          board_id,
          image_names,
        })
      );
    },
  });

  // batch selection removed from board
  startAppListening({
    actionCreator: batchSelectionRemovedFromBoard,
    effect: (action, { getState, dispatch }) => {
      const image_names = getState().batch.selection;
      dispatch(
        boardImagesApi.endpoints.removeManyImagesFromBoard.initiate(image_names)
      );
    },
  });

  // many images added to board - fulfilled
  startAppListening({
    matcher: boardImagesApi.endpoints.addManyImagesToBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_names } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_names } },
        'Images added to board'
      );

      const updates = image_names.map((image_name) => ({
        id: image_name,
        changes: { board_id },
      }));

      dispatch(imageUpdatedMany(updates));
    },
  });

  // many images added to board - rejected
  startAppListening({
    matcher: boardImagesApi.endpoints.addManyImagesToBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_names } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_names } },
        'Problem adding many images to board'
      );
    },
  });

  // remove many images from board - fulfilled
  startAppListening({
    matcher: boardImagesApi.endpoints.removeManyImagesFromBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const image_names = action.meta.arg.originalArgs;

      moduleLog.debug({ data: { image_names } }, 'Images removed from board');

      const updates = image_names.map((image_name) => ({
        id: image_name,
        changes: { board_id: undefined },
      }));

      dispatch(imageUpdatedMany(updates));
    },
  });

  // remove many images from board - rejected
  startAppListening({
    matcher: boardImagesApi.endpoints.removeManyImagesFromBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const image_names = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { image_names } },
        'Problem removing many images from board'
      );
    },
  });
};

export const gallerySelectionAddedToBoard = createAction<{ board_id: string }>(
  'boards/gallerySelectionAddedToBoard'
);

export const gallerySelectionRemovedFromBoard = createAction(
  'boards/gallerySelectionAddedToBoard'
);

export const batchSelectionAddedToBoard = createAction<{
  board_id: string;
}>('boards/batchSelectionAddedToBoard');

export const batchSelectionRemovedFromBoard = createAction(
  'boards/batchSelectionAddedToBoard'
);
