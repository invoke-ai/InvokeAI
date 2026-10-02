import { GenerateCanvasCompositingSection } from './GenerateCanvasCompositingSection';
import { GenerateCanvasScalingSection } from './GenerateCanvasScalingSection';
import { GenerateDenoisingStrength } from './GenerateDenoisingStrength';

/** The Generate form's canvas-only sections: denoising strength, scale before processing, then compositing. */
export const GenerateCanvasSections = () => (
  <>
    <GenerateDenoisingStrength />
    <GenerateCanvasScalingSection />
    <GenerateCanvasCompositingSection />
  </>
);
