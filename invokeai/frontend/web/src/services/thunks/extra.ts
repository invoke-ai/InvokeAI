import { createAppAsyncThunk } from 'app/storeUtils';
import { addImage } from 'features/gallery/store/gallerySlice';
import { forEach } from 'lodash';
import { SessionsService } from 'services/api';
import { isImageOutput } from 'services/types/guards';
import { v4 as uuidv4 } from 'uuid';

type GetGalleryImagesArg = {
  count: number;
};

/**
 * Get the last 20 sessions' worth of images.
 *
 * This should be at most 20 images so long as we continue to make a new session for every
 * generation.
 *
 * If a session was created but no image generated, this will be < 20 images.
 *
 * When we allow more images per sesssion, this is kinda no longer a viable way to grab results,
 * because a session could have many, many images. In that situation, barring a change to the api,
 * we have to keep track of images we've grabbed and the session they came from, so that when we
 * want to load more, we can "resume" fetching images from that session.
 */
export const getGalleryImages = createAppAsyncThunk(
  'api/getGalleryImages',
  async (arg: GetGalleryImagesArg, { dispatch }) => {
    const response = await SessionsService.listSessions({
      page: 0,
      perPage: 20,
    });

    response.items.forEach((session) => {
      forEach(session.results, (result) => {
        if (isImageOutput(result)) {
          const url = `api/v1/images/${result.image.image_type}/${result.image.image_name}`;

          dispatch(
            addImage({
              category: 'result',
              image: {
                uuid: uuidv4(),
                url,
                thumbnail: '',
                width: 512,
                height: 512,
                category: 'result',
                name: result.image.image_name,
                mtime: new Date().getTime(),
              },
            })
          );
        }
      });
    });
  }
);
