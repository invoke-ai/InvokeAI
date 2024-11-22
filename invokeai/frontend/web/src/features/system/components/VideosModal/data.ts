/**
 * To add a support video, you'll need to add the video to the list below.
 *
 * The `tKey` is a sub-key in the translation file `invokeai/frontend/web/public/locales/en.json`.
 * Add the title and description under `supportVideos.videos`, following the existing format.
 */

export type VideoData = {
  tKey: string;
  link: string;
  length: {
    minutes: number;
    seconds: number;
  };
};

export const gettingStartedVideos: VideoData[] = [
  {
    tKey: 'creatingYourFirstImage',
    link: 'https://www.youtube.com/watch?v=jVi2XgSGrfY&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&index=1&t=29s&pp=iAQB',
    length: { minutes: 6, seconds: 0 },
  },
  {
    tKey: 'usingControlLayersAndReferenceGuides',
    link: 'https://www.youtube.com/watch?v=crgw6bEgyrw&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&index=2&t=70s&pp=iAQB',
    length: { minutes: 5, seconds: 30 },
  },
  {
    tKey: 'understandingImageToImageAndDenoising',
    link: 'https://www.youtube.com/watch?v=tvj8-0s6S2U&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&index=3&t=1s&pp=iAQB',
    length: { minutes: 2, seconds: 37 },
  },
  {
    tKey: 'exploringAIModelsAndConceptAdapters',
    link: 'https://www.youtube.com/watch?v=iwBmBQMZ0UA&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&index=4&pp=iAQB',
    length: { minutes: 8, seconds: 52 },
  },
  {
    tKey: 'creatingAndComposingOnInvokesControlCanvas',
    link: 'https://www.youtube.com/watch?v=O4LaFcYFxlA',
    length: { minutes: 2, seconds: 52 },
  },
  {
    tKey: 'upscaling',
    link: 'https://www.youtube.com/watch?v=OCb19_P0nro&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&index=6&t=2s&pp=iAQB',
    length: { minutes: 4, seconds: 0 },
  },
];

export const controlCanvasVideos: VideoData[] = [
  {
    tKey: 'howDoIGenerateAndSaveToTheGallery',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=dbjmBc1iDAUpE1k5&t=26',
    length: { minutes: 0, seconds: 49 },
  },
  {
    tKey: 'howDoIEditOnTheCanvas',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=U_bFl9HsvSuejbxp&t=76',
    length: { minutes: 0, seconds: 58 },
  },
  {
    tKey: 'howDoIDoImageToImageTransformation',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=fjhTeY-yZ3qsEzEM&t=138',
    length: { minutes: 0, seconds: 51 },
  },
  {
    tKey: 'howDoIUseControlNetsAndControlLayers',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=x5KcYvkHbvR9ifsX&t=192',
    length: { minutes: 1, seconds: 41 },
  },
  {
    tKey: 'howDoIUseGlobalIPAdaptersAndReferenceImages',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=O940rNHiHGKXknK2&t=297',
    length: { minutes: 0, seconds: 43 },
  },
  {
    tKey: 'howDoIUseInpaintMasks',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=3DZhmerkzUmvJJSn&t=345',
    length: { minutes: 1, seconds: 9 },
  },
  {
    tKey: 'howDoIOutpaint',
    link: 'https://youtu.be/Tl-69JvwJ2s?si=IIwkGZLq1PfLf80Q&t=420',
    length: { minutes: 0, seconds: 48 },
  },
];

export const studioSessionsPlaylistLink = 'https://www.youtube.com/playlist?list=PLvWK1Kc8iXGq_8tWZqnwDVaf9uhlDC09U';
