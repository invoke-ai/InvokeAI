import { useStore } from '@nanostores/react';
import { $authToken } from 'app/store/nanostores/authToken';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageUploadedClientSide } from 'features/gallery/store/actions';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { useCallback } from 'react';
import { useCreateImageUploadEntryMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

type PresignedUrlResponse = {
  fullUrl: string;
  thumbnailUrl: string;
};

const isPresignedUrlResponse = (response: unknown): response is PresignedUrlResponse => {
  return typeof response === 'object' && response !== null && 'fullUrl' in response && 'thumbnailUrl' in response;
};

export const useClientSideUpload = () => {
  const dispatch = useAppDispatch();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const authToken = useStore($authToken);
  const [createImageUploadEntry] = useCreateImageUploadEntryMutation();

  const clientSideUpload = useCallback(
    async (file: File, i: number): Promise<ImageDTO> => {
      const image = new Image();
      const objectURL = URL.createObjectURL(file);
      image.src = objectURL;
      let width = 0;
      let height = 0;
      let thumbnail: Blob | undefined;

      await new Promise<void>((resolve) => {
        image.onload = () => {
          width = image.naturalWidth;
          height = image.naturalHeight;

          // Calculate thumbnail dimensions maintaining aspect ratio
          let thumbWidth = width;
          let thumbHeight = height;
          if (width > height && width > 256) {
            thumbWidth = 256;
            thumbHeight = Math.round((height * 256) / width);
          } else if (height > 256) {
            thumbHeight = 256;
            thumbWidth = Math.round((width * 256) / height);
          }

          const canvas = document.createElement('canvas');
          canvas.width = thumbWidth;
          canvas.height = thumbHeight;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(image, 0, 0, thumbWidth, thumbHeight);

          canvas.toBlob(
            (blob) => {
              if (blob) {
                thumbnail = blob;
                // Clean up resources
                URL.revokeObjectURL(objectURL);
                image.src = ''; // Clear image source
                image.remove(); // Remove the image element
                canvas.width = 0; // Clear canvas
                canvas.height = 0;
                resolve();
              }
            },
            'image/webp',
            0.8
          );
        };

        // Handle load errors
        image.onerror = () => {
          URL.revokeObjectURL(objectURL);
          image.remove();
          resolve();
        };
      });
      const { presigned_url, image_dto } = await createImageUploadEntry({
        width,
        height,
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      }).unwrap();

      const response = await fetch(presigned_url, {
        method: 'GET',
        ...(authToken && {
          headers: {
            Authorization: `Bearer ${authToken}`,
          },
        }),
      }).then((res) => res.json());

      if (!isPresignedUrlResponse(response)) {
        throw new Error('Invalid response');
      }

      const fullUrl = response.fullUrl;
      const thumbnailUrl = response.thumbnailUrl;

      await fetch(fullUrl, {
        method: 'PUT',
        body: file,
      });

      await fetch(thumbnailUrl, {
        method: 'PUT',
        body: thumbnail,
      });

      dispatch(imageUploadedClientSide({ imageDTO: image_dto, silent: false, isFirstUploadOfBatch: i === 0 }));

      return image_dto;
    },
    [autoAddBoardId, authToken, createImageUploadEntry, dispatch]
  );

  return clientSideUpload;
};
