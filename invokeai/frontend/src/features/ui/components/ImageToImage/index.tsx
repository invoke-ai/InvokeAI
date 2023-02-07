import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import ImageToImageDisplay from './ImageToImageDisplay';
import ImageToImagePanel from './ImageToImagePanel';

export default function ImageToImageWorkarea() {
  return (
    <InvokeWorkarea optionsPanel={<ImageToImagePanel />}>
      <ImageToImageDisplay />
    </InvokeWorkarea>
  );
}
