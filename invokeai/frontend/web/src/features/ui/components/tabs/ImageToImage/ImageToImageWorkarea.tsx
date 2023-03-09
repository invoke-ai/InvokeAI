import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import ImageToImageContent from './ImageToImageContent';
import ImageToImageParameters from './ImageToImageParameters';

export default function ImageToImageWorkarea() {
  return (
    <InvokeWorkarea parametersPanelContent={<ImageToImageParameters />}>
      <ImageToImageContent />
    </InvokeWorkarea>
  );
}
