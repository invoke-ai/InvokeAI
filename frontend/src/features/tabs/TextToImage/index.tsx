import TextToImagePanel from './TextToImagePanel';
import InvokeWorkarea from 'features/tabs/InvokeWorkarea';
import TextToImageDisplay from './TextToImageDisplay';

export default function TextToImageWorkarea() {
  return (
    <InvokeWorkarea optionsPanel={<TextToImagePanel />}>
      <TextToImageDisplay />
    </InvokeWorkarea>
  );
}
