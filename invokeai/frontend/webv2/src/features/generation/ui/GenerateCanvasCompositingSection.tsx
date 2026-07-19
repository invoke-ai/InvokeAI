import { useGenerationUi } from './GenerationUiContext';

export const GenerateCanvasCompositingSection = () => {
  const { CanvasCompositingSection, project } = useGenerationUi();
  return project.invocationSourceId === 'canvas' ? <CanvasCompositingSection /> : null;
};
