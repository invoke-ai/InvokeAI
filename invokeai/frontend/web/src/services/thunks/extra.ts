import { createAppAsyncThunk } from 'app/storeUtils';
import { map } from 'lodash';
import { SessionsService } from 'services/api';
import { isImageOutput } from 'services/types/guards';
import { buildImageUrls } from 'services/util/buildImageUrls';
import { extractTimestampFromResultImageName } from 'services/util/extractTimestampFromResultImageName';
import { resultsReceived } from 'features/gallery/store/resultsSlice';

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

    // build flattened array of results ojects, use lodash `map()` to make results object an array
    const allResults = response.items.flatMap((session) =>
      map(session.results)
    );

    // filter out non-image-outputs (eg latents, prompts, etc)
    const imageOutputResults = allResults.filter(isImageOutput);

    // build ResultImage objects
    const resultImages = imageOutputResults.map((result) => {
      const name = result.image.image_name;

      const { imageUrl, thumbnailUrl } = buildImageUrls('results', name);
      const timestamp = extractTimestampFromResultImageName(name);

      return {
        name,
        url: imageUrl,
        thumbnail: thumbnailUrl,
        timestamp,
        height: 512,
        width: 512,
      };
    });

    // update the results slice
    dispatch(resultsReceived(resultImages));

    // response.items.forEach((session) => {
    //   forEach(session.results, (result) => {
    //     if (isImageOutput(result)) {
    //       const { imageUrl, thumbnailUrl } = buildImageUrls(
    //         result.image.image_type!, // fix the generated types to avoid non-null assertion
    //         result.image.image_name! // fix the generated types to avoid non-null assertion
    //       );

    //       dispatch

    //       dispatch(
    //         addImage({
    //           category: 'result',
    //           image: {
    //             uuid: uuidv4(),
    //             url: imageUrl,
    //             thumbnail: ,
    //             width: 512,
    //             height: 512,
    //             category: 'result',
    //             name: result.image.image_name,
    //             mtime: new Date().getTime(),
    //           },
    //         })
    //       );
    //     }
    //   });
    // });
  }
);
