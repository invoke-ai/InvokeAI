import { useGenerationUi } from './GenerationUiContext';

export const GenerateCanvasSections = () => {
  const { CanvasGenerationSections, project } = useGenerationUi();
  return project.invocationSourceId === 'canvas' ? <CanvasGenerationSections /> : null;
};
