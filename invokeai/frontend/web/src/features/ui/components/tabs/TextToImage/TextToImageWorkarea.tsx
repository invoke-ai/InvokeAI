import InvokeWorkarea from 'features/ui/components/common/InvokeWorkarea';
import TextToImageContent from './TextToImageContent';
import TextToImageParameters from './TextToImageParameters';

export default function TextToImageWorkarea() {
  return (
    <InvokeWorkarea parametersPanel={<TextToImageParameters />}>
      <TextToImageContent />
    </InvokeWorkarea>
  );
}
