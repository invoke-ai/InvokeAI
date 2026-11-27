import { GenerateCanvasCompositingSection } from './GenerateCanvasCompositingSection';

// The Generate form's canvas slots load together, as one chunk.
export { GenerateCanvasRenderSize } from './GenerateCanvasRenderSize';
export { GenerateDenoisingStrength } from './GenerateDenoisingStrength';

export const GenerateCanvasSections = () => <GenerateCanvasCompositingSection />;
