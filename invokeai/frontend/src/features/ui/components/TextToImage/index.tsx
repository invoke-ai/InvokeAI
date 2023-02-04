import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import TextToImageDisplay from './TextToImageDisplay';
import TextToImagePanel from './TextToImagePanel';

export default function TextToImageWorkarea() {
  return (
    <InvokeWorkarea optionsPanel={<TextToImagePanel />}>
      <TextToImageDisplay />
    </InvokeWorkarea>
  );
}
