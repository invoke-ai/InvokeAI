const GUIDANCE_SLIDER_SPAN = 10;
const GUIDANCE_INPUT_MAX = 100;

/**
 * Slider track for the guidance field. The input accepts values up to the architecture ceiling; the track spans
 * `GUIDANCE_SLIDER_SPAN` above zero, or above the floor when the floor already reaches that span (FLUX Fill), so the
 * track never collapses onto values the model refuses.
 */
export const getGuidanceSliderMax = (guidanceMin: number, guidanceMax: number | null): number => {
  const inputMax = guidanceMax ?? GUIDANCE_INPUT_MAX;
  const trackEnd = guidanceMin >= GUIDANCE_SLIDER_SPAN ? guidanceMin + GUIDANCE_SLIDER_SPAN : GUIDANCE_SLIDER_SPAN;

  return Math.max(guidanceMin, Math.min(trackEnd, inputMax));
};

/** Upper bound of the typed input for the guidance field. */
export const getGuidanceInputMax = (guidanceMax: number | null): number => guidanceMax ?? GUIDANCE_INPUT_MAX;
