import { useGenerationUi } from './GenerationUiContext';

export const GenerateCanvasCompositingSection = () => {
  const { CanvasCompositingSection, invocationSourceId } = useGenerationUi();
  return invocationSourceId === 'canvas' ? <CanvasCompositingSection /> : null;
};
