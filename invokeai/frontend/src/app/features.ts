import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type FeatureHelpInfo = {
  text: string;
  href: string;
  guideImage: string;
};

export enum Feature {
  PROMPT,
  GALLERY,
  OTHER,
  SEED,
  VARIATIONS,
  UPSCALE,
  FACE_CORRECTION,
  IMAGE_TO_IMAGE,
  BOUNDING_BOX,
  SEAM_CORRECTION,
  INFILL_AND_SCALING,
}
/** For each tooltip in the UI, the below feature definitions & props will pull relevant information into the tooltip.
 *
 * To-do: href & GuideImages are placeholders, and are not currently utilized, but will be updated (along with the tooltip UI) as feature and UI develop and we get a better idea on where things "forever homes" will be .
 */
const useFeatures = (): Record<Feature, FeatureHelpInfo> => {
  const { t } = useTranslation();
  return useMemo(
    () => ({
      [Feature.PROMPT]: {
        text: t('tooltip.feature.prompt'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.GALLERY]: {
        text: t('tooltip.feature.gallery'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.OTHER]: {
        text: t('tooltip.feature.other'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.SEED]: {
        text: t('tooltip.feature.seed'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.VARIATIONS]: {
        text: t('tooltip.feature.variations'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.UPSCALE]: {
        text: t('tooltip.feature.upscale'),
        href: 'link/to/docs/feature1.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.FACE_CORRECTION]: {
        text: t('tooltip.feature.faceCorrection'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.IMAGE_TO_IMAGE]: {
        text: t('tooltip.feature.imageToImage'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.BOUNDING_BOX]: {
        text: t('tooltip.feature.boundingBox'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.SEAM_CORRECTION]: {
        text: t('tooltip.feature.seamCorrection'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
      [Feature.INFILL_AND_SCALING]: {
        text: t('tooltip.feature.infillAndScaling'),
        href: 'link/to/docs/feature3.html',
        guideImage: 'asset/path.gif',
      },
    }),
    [t]
  );
};

export const useFeatureHelpInfo = (feature: Feature): FeatureHelpInfo => {
  const features = useFeatures();
  return features[feature];
};
