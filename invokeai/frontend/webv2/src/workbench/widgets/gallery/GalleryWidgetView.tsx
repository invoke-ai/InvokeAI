import { useEffect, useState } from 'react';
import { PiImageBold } from 'react-icons/pi';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import {
  createGalleryBoard,
  listGalleryBoards,
  listGalleryImages,
  type GalleryBoard,
  type GalleryImage,
} from '../../gallery/api';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';
import { GalleryPanelContent } from './GalleryPanelContent';
import {
  getGalleryRecentImagesKey,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
  getGalleryStateView,
  getGalleryView,
} from './galleryStateView';

interface BackendImagesResult {
  images: GalleryImage[];
  queryKey: string;
}

const EMPTY_BACKEND_IMAGES: GalleryImage[] = [];

const getGalleryQueryKey = ({
  boardId,
  galleryView,
  searchTerm,
}: {
  boardId: string;
  galleryView: string;
  searchTerm: string;
}): string => `${galleryView}\0${boardId}\0${searchTerm.trim()}`;

export const GalleryWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const [backendBoards, setBackendBoards] = useState<GalleryBoard[]>([]);
  const [backendImagesResult, setBackendImagesResult] = useState<BackendImagesResult | null>(null);
  const [loadingImageQueryKey, setLoadingImageQueryKey] = useState<string | null>(null);
  const galleryValues = activeProject.widgetStates.gallery.values;
  const selectedBoardId = getGallerySelectedBoardId(galleryValues, backendBoards);
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const imageQueryKey = getGalleryQueryKey({ boardId: selectedBoardId, galleryView, searchTerm });
  const isAwaitingCurrentImages = backendImagesResult !== null && backendImagesResult.queryKey !== imageQueryKey;
  const backendImages =
    backendImagesResult === null
      ? null
      : backendImagesResult.queryKey === imageQueryKey
        ? backendImagesResult.images
        : EMPTY_BACKEND_IMAGES;
  const isLoadingImages = loadingImageQueryKey === imageQueryKey || isAwaitingCurrentImages;
  const gallery = getGalleryStateView(galleryValues, backendBoards, backendImages, isLoadingImages);
  const isWidePlacement = region === 'center' || (region === 'bottom' && presentation === 'expanded');

  useEffect(() => {
    let isStale = false;

    listGalleryBoards()
      .then((boards) => {
        if (!isStale) {
          setBackendBoards(boards);
        }
      })
      .catch((error: unknown) => {
        dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });
      });

    return () => {
      isStale = true;
    };
  }, [dispatch, recentImagesKey]);

  useEffect(() => {
    let isStale = false;

    setLoadingImageQueryKey(imageQueryKey);
    listGalleryImages({
      boardId: selectedBoardId,
      galleryView,
      searchTerm,
    })
      .then((images) => {
        if (!isStale) {
          setBackendImagesResult({ images, queryKey: imageQueryKey });
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          setBackendImagesResult({ images: [], queryKey: imageQueryKey });
          dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });
        }
      })
      .finally(() => {
        if (!isStale) {
          setLoadingImageQueryKey((currentQueryKey) => (currentQueryKey === imageQueryKey ? null : currentQueryKey));
        }
      });

    return () => {
      isStale = true;
    };
  }, [dispatch, galleryView, imageQueryKey, recentImagesKey, searchTerm, selectedBoardId]);

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={PiImageBold}>Gallery: {gallery.images.length}</StatusWidgetChip>;
  }

  return (
    <GalleryPanelContent
      gallery={gallery}
      layout={isWidePlacement ? 'wide' : 'stacked'}
      region={region}
      onCreateBoard={() => {
        createGalleryBoard(`Board ${backendBoards.length}`)
          .then((board) => {
            setBackendBoards((boards) => [...boards, board]);
            dispatch({ boardId: board.id, type: 'selectGalleryBoard' });
          })
          .catch((error: unknown) => {
            dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });
          });
      }}
      onSearch={(searchTerm) => dispatch({ searchTerm, type: 'setGallerySearchTerm' })}
      onSetImageDensityPercent={(imageDensityPercent) =>
        dispatch({ imageDensityPercent, type: 'setGalleryImageDensityPercent' })
      }
      onSelectBoard={(boardId) => dispatch({ boardId, type: 'selectGalleryBoard' })}
      onSelectImage={(image) => dispatch({ image, type: 'selectGalleryImage' })}
      onSetView={(galleryView) => dispatch({ galleryView, type: 'setGalleryView' })}
    />
  );
};
