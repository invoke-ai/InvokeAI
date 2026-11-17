import { GenerationSectionJumpMenu } from '@features/generation/components';

export const UpscaleHeaderActions = ({ instance }: { instance?: { id: string } }) => (
  <GenerationSectionJumpMenu panelId={instance?.id} />
);
