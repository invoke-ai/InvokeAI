import { GenerateCanvasCompositingSection } from './GenerateCanvasCompositingSection';
import { GenerateDenoisingStrength } from './GenerateDenoisingStrength';

// The Generate form's canvas slots load together, as one chunk.
export { GenerateCanvasRenderSize } from './GenerateCanvasRenderSize';

/** The Generate form's canvas-only sections: denoising strength, then compositing. */
export const GenerateCanvasSections = () => (
  <>
    <GenerateDenoisingStrength />
    <GenerateCanvasCompositingSection />
  </>
);
