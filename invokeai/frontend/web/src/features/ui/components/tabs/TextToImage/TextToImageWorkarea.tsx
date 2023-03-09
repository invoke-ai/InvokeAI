import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import TextToImageContent from './TextToImageContent';
import TextToImageParameters from './TextToImageParameters';

export default function TextToImageWorkarea() {
  return (
    <InvokeWorkarea parametersPanelContent={<TextToImageParameters />}>
      <TextToImageContent />
    </InvokeWorkarea>
  );
}
